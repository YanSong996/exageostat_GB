/**
 *
 * Copyright (c) 2017-2019  King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * ExaGeoStat is a software package provided by KAUST
 **/
/**
 *
 * @file zgen_mle_test.c
 *
 * A complete example to test ExaGeoStat supported function (i.e., dataset generator, Maximum Likelihood Function (MLE), Prediction)
 *
 * @version 1.0.0
 *
 * @author Sameh Abdulah
 * @date 2019-08-06
 *
 **/
#include "examples.h"
#include "../src/include/MLE.h"

int main(int argc, char **argv) {

	//Values for the mean trend removal
	int M=6;
	int T=24*365;  //Fixed
	int no_years= 751;
	//************************************


	//initialization
	double *starting_theta;
	double *target_theta;
	double *initial_theta;  //for testing case
	int num_params = 0;
	int N, dts, lts, log;
	int i = 0;
	int p = 0;         //univariate/bivariate/multivariate
	int zvecs = 1, nZmiss = 0, test = 0, gpus = 0;
	double x_max, x_min, y_max, y_min;
	int p_grid, q_grid, ncores;
	double  opt_f;
	arguments arguments;
	double all_time=0.0;
	double pred_time=0.0;
	nlopt_opt opt;
	MLE_data data;
	int seed = 0;
	location *locations = NULL;
	double prediction_error = 0.0;
	//Arguments default values
	set_args_default(&arguments);
	argp_parse(&argp, argc, argv, 0, 0, &arguments);
	check_args(&arguments);

	if(strcmp(arguments.kernel_fun, "trend_model")   == 0)
	{
		num_params = 1;
		p          = 1;
	}
	else
	{
		fprintf(stderr,"Choosen kernel is not exist(just trend_model for this example)!\n");
		fprintf(stderr, "Called function is: %s\n",__func__);
		exit(0);
	}

	double* lb = (double *) malloc(num_params * sizeof(double));
	double* up = (double *) malloc(num_params * sizeof(double));
	int iseed[4]={seed, seed, seed, 1};

	//Memory allocation
	starting_theta    = (double *) malloc(num_params * sizeof(double));
	initial_theta    = (double *) malloc(num_params * sizeof(double));
	target_theta    = (double *) malloc(num_params * sizeof(double));

	//MLE_data data initialization
	init(&test, &N,  &ncores,
			&gpus, &p_grid, &q_grid,
			&zvecs, &dts, &lts,
			&nZmiss, &log, initial_theta, 
			starting_theta, target_theta, lb,
			up, &data, &arguments);


	//read forcing data
	double * forcing = (double *) malloc(751 * sizeof(double));
	//Remove it produce Seg fault!!!!
	int* t2m =(double *) malloc(1 * sizeof(int));
	forcing = readObsFile("/scratch/abdullsm/GB24_stage0/exageostat_GB/forcing_new.csv", 751);
	fprintf(stderr, "focring: %0.16f %e\n", forcing[0], forcing[1]);
	data.forcing=forcing;
	fprintf(stderr, "data.focring: %f %f\n", data.forcing[0], data.forcing[1]);
	//********************************* Read form the NetCDF file
	int retval;
	size_t  t2m_len;
	int  lon_varid, lat_varid, time_varid,t2m_varid;
	size_t lat_len, lon_len, time_len;
	int 	len;
	int ncid;

	//to be remove
	int v= 8760;//*no_years;
	N =8760 *61;// N =time_len;
	data.M= M;
	data.T=T;
	data.no_years=no_years;
//	t2m = (short int *) malloc(v *  sizeof(int));
	//*****************************
	//
	exageostat_init(&ncores, &gpus, &dts, &lts);
	// Optimizater initialization
	//NLOPT_LN_BOBYQA
	opt=nlopt_create(NLOPT_LN_BOBYQA, num_params);
	init_optimizer(&opt, lb, up, pow(10, -1.0 * data.opt_tol));
	nlopt_set_maxeval(opt, data.opt_max_iters);
	fprintf(stderr, "data.forcing: %f %f\n", data.forcing[0], data.forcing[1]);
	data.mean_trend = 1;
	if(strcmp (data.computation, "exact") == 0)
		MORSE_dmle_Call(&data, ncores, gpus, dts, p_grid, q_grid, N,  0, 0);

	print_summary(test, N, ncores, gpus, dts, lts,  data.computation, zvecs, p_grid, q_grid, data.precision);

	char numStr[20];
	char path[150] = "/scratch/abdullsm/ERA_data/data_1941.nc";

	ncid 	= openFileNC(&data, path);
	fprintf(stderr, "%s\n", path);

	//nZmiss          = countlinesNC(ncid, dim1, dim2);
	if ((retval = nc_inq_dimid(ncid,"longitude",&lon_varid)))
		printf("Error: %s\n", nc_strerror(retval));
	if ((retval = nc_inq_dimlen(ncid, lon_varid, &lon_len)))
		printf("Error: %s\n", nc_strerror(retval));
	//**********
	if ((retval = nc_inq_dimid(ncid,"latitude",&lat_varid)))
		printf("Error: %s\n", nc_strerror(retval));
	if ((retval = nc_inq_dimlen(ncid, lat_varid, &lat_len)))
		printf("Error: %s\n", nc_strerror(retval));
	//*********
	if ((retval = nc_inq_dimid(ncid,"time",&time_varid)))
		printf("Error: %s\n", nc_strerror(retval));
	if ((retval = nc_inq_dimlen(ncid, time_varid, &time_len)))
		printf("Error: %s\n", nc_strerror(retval));

	len = lon_len * lat_len * time_len;
	fprintf(stderr, "%d, %d, %d\n", lon_varid, lat_varid, time_varid);	
	fprintf(stderr, "%d, %d, %d, %d\n", lon_len, lat_len, time_len, len);
	if ((retval = nc_inq_varid(ncid, "t2m", &t2m_varid)))
		printf("Error: %s\n", nc_strerror(retval));
	//fprintf(stderr, "=====t2m_varid=%d, \n", t2m_varid);
	//

	int location = 0;
	double* t2m_temp = (double *) malloc(v * (2000-1940) *  sizeof(double));
	for (int l=0;l<1440; l++)
		for (int u=0;u<721; u++)
		{
			size_t index[] = {0, u, l};
				if( l==439  && u ==185)
		//	if( true)
			{
				int r=0;
				int  ll = 0;
				double sum_obs=0;
				double  scaling_var, offset_var;
				for(int y=1940;y<2000;y++)
				{
					fprintf(stderr, "================================ \n");
					// Convert integer to string
					char path2[150] = "/scratch/abdullsm/ERA_data/data_";
					sprintf(numStr, "%d", y);
					strcat(path2 , numStr);
					strcat(path2, ".nc");
					ncid    = openFileNC(&data, path2);
					fprintf(stderr, "%s\n", path2);
					if(y %4 ==0)
					{		t2m = (short int *) malloc((v+24) *  sizeof(int));
						time_len=v+24;
					}
					else
					{	t2m = (short int *) malloc(v *  sizeof(int));
						time_len=v;
					}
					//		fprintf(stderr, "longitude: %d latitude: %d \n", l, u); 
					size_t count[] = {time_len, 1, 1};
					//		fprintf(stderr, "longitude: %d latitude: %d \n", l, u);
					if (retval = nc_get_vara_int(ncid, t2m_varid, index, count, t2m))
						printf("Error: %s\n", nc_strerror(retval));	
					//		fprintf(stderr, "longitude: %d latitude: %d \n", l, u);
					if (retval =nc_get_att_double(ncid, t2m_varid, "scale_factor", &scaling_var))
						printf("Error: %s\n", nc_strerror(retval));
					//		fprintf(stderr, "longitude: %d latitude: %d \n", l, u);
					if (retval =nc_get_att_double(ncid, t2m_varid, "add_offset", &offset_var))
						printf("Error: %s\n", nc_strerror(retval));
					//		fprintf(stderr, "longitude: %d latitude: %d \n", l, u);
					//		fprintf(stderr, "scaling_var: %e -- offset_var: %e\n", scaling_var, offset_var);

					//to be removed
					// 8784
					if(y%4==0)
						for(int k=0;k<v+24;k++)
						{
							if(k>= 1416 && k<=1439);
							else
							{	t2m_temp[ll++]= ((double)t2m[k]*scaling_var) + offset_var - 273.15;
								//		sum_obs+=((double)t2m[k]*scaling_var) + offset_var - 273.15;
							}
							//if(k>v)
							//	fprintf(stderr, "\nt2m_temp[k]: %d, %f\n", t2m[k], t2m_temp[k]);
							//      exit(0);
							//      t2m_temp[k]= t2m_temp[k] * 0.00203215878891646
						}
					else

						for(int k=0;k<v;k++)
						{
							if( ll==0)
								fprintf(stderr, "==================>\n");
							t2m_temp[ll++]= ((double)t2m[k]*scaling_var) + offset_var - 273.15;
							//	fprintf(stderr, "\nt2m_temp[k]: %d, %f\n", t2m[k], t2m_temp[k]);
							//	exit(0);
							//	t2m_temp[k]= t2m_temp[k] * 0.00203215878891646
						}
					//*****************************
					//fprintf(stderr, "sum: ---- %f\n", sum_obs);
					//	exit(0);
					//    for(int i=0;i<5;i++)
					//  fprintf(stderr, "%f\n", t2m_temp[i]);
					// exit(0);
					closeFileNC(&data, ncid);//	free (path2);
				}
				//	fprintf(stderr, "888888888longitude: %d latitude: %d \n", l, u);
				MORSE_MLE_dzcpy(&data, t2m_temp);
				//	fprintf(stderr, "99999999999longitude: %d latitude: %d \n", l, u);
				START_TIMING(data.total_exec_time);
				nlopt_set_max_objective(opt, MLE_alg, (void *)&data);
				starting_theta[0] = 0.9;
				nlopt_optimize(opt, starting_theta, &opt_f);
				STOP_TIMING(data.total_exec_time);
				//	starting_theta[0] = 0.272049;
				fprintf(stderr, "final theta: %f\n", starting_theta[0]);

				mean_trend(starting_theta, &data, location);
				location++;
				exit(0);
			}
			//fprintf(stderr, "longitude: %d latitude: %d \n", l, u);
		}

	//Phase(2)
	//	int L = 720;
	//	if(strcmp (data.computation, "exact") == 0)
	//		MORSE_dmle_Call_phase_2(&data, ncores, gpus, dts, p_grid, q_grid, N,  L);

	//	read_csv_files();
	//
	//	phase_2_calcs();




	exit(0);

	MORSE_Finalize();
	return 0;
}

