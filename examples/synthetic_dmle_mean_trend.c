/**
 *
 * Copyright (c) 2017-2024  King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * ExaGeoStat is a software package provided by KAUST
 **/
/**
 *
 * @file synthetic_dmle_mean_trend.c
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
#define VERBOSE(str)    \
    if (MORSE_My_Mpi_Rank() == 0){        \
        fprintf(stdout, "%s", str);     \
        fflush(stdout);\
    }

int isLeapYear(int year) {
    if (year % 400 == 0) {
        return 1;
    } else if (year % 100 == 0) {
        return 0;
    } else if (year % 4 == 0) {
        return 1;
    } else {
        return 0;
    }
}

int main(int argc, char **argv) {

	//Values for the mean trend removal
	int M=10;
	int T=365*24;  //Fixed
	int no_years= 751;
	//************************************


	//initialization
	double *starting_theta;
	double *target_theta;
	double *initial_theta;  //for testing case
	int num_params = 0;
	size_t N, dts, lts, log;
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

	fprintf(stderr, "Number of params to optimize: %d\n", num_params);
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
	int* t2m;// =(double *) malloc(1 * sizeof(int));
	int* t2m_local;// =(double *) malloc(1 * sizeof(int));
	forcing = readObsFile("/scratch/abdullsm/GB24_stage0/exageostat_GB/forcing_new.csv", 751);
	fprintf(stderr, "forcing: %0.16f %e\n", forcing[0], forcing[1]);
	data.forcing=forcing;
	fprintf(stderr, "data.forcing: %f %f\n", data.forcing[0], data.forcing[1]);
	//********************************* Read form the NetCDF file
	int retval;
	size_t  t2m_len;
	int  lon_varid, lat_varid, time_varid,t2m_varid;
	size_t lat_len, lon_len, time_len;
	int 	len;
	int ncid;

	//to be remove
	int v= 365;//*no_years;
	N = 365 * 24* (2023-1988);// N =time_len;
	data.M= M;
	data.T=T;
	data.no_years=no_years;
	//*****************************
	exageostat_init(&ncores, &gpus, &dts, &lts);
	// Optimizater initialization
	opt=nlopt_create(NLOPT_LN_BOBYQA, num_params);
	init_optimizer(&opt, lb, up, pow(10, -1.0 * data.opt_tol));
	nlopt_set_maxeval(opt, 30);
	fprintf(stderr, "data.forcing: %f %f\n", data.forcing[0], data.forcing[1]);
	data.mean_trend = 1;
	print_summary(test, N, ncores, gpus, dts, lts,  data.computation, zvecs, p_grid, q_grid, data.precision);

	char numStr[20];
	char path[150] = "/scratch/abdullsm/GB24_stage0/exageostat_GB/ERA_data/data_1945.nc";

	ncid 	= openFileNCmpi(&data, path);
	fprintf(stderr, "%s\n", path);

	if ((retval = ncmpi_inq_dimid(ncid,"longitude",&lon_varid)))
		printf("Error: %s\n", nc_strerror(retval));
	if ((retval = ncmpi_inq_dimlen(ncid, lon_varid, &lon_len)))
		printf("Error: %s\n", nc_strerror(retval));
	//**********
	if ((retval = ncmpi_inq_dimid(ncid,"latitude",&lat_varid)))
		printf("Error: %s\n", nc_strerror(retval));
	if ((retval = ncmpi_inq_dimlen(ncid, lat_varid, &lat_len)))
		printf("Error: %s\n", nc_strerror(retval));
	//*********
	if ((retval = ncmpi_inq_dimid(ncid,"time",&time_varid)))
		printf("Error: %s\n", nc_strerror(retval));
	if ((retval = ncmpi_inq_dimlen(ncid, time_varid, &time_len)))
		printf("Error: %s\n", nc_strerror(retval));


	len = lon_len * lat_len * time_len;
	fprintf(stderr, "%d, %d, %d\n", lon_varid, lat_varid, time_varid);	
	fprintf(stderr, "%d, %d, %d, %d\n", lon_len, lat_len, time_len, len);
	if ((retval = ncmpi_inq_varid(ncid, "t2m", &t2m_varid)))
		printf("Error: %s\n", nc_strerror(retval));


	int no_locs=1440;
//int no_locs=10;
	double *t2m_hourly_per_year[no_locs];
	int t2m_hourly_per_year_count[no_locs];
	for(int k=0;k<no_locs;k++)
	{
		t2m_hourly_per_year[k] = (double *) malloc(v * 24* (2023-1988)  *   sizeof(double));
		if(t2m_hourly_per_year[k] == NULL)
			fprintf(stderr, "no enough memory... t2m_hourly_per_year");
		t2m_hourly_per_year_count[k]=0;
	}

	int rank;
	int nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	int min_loc=lts;
	int max_loc= data.hicma_maxrank;
	int r=0;
	double sum_obs=0;
	double  scaling_var, offset_var;

	for(int y=1988;y<2023;y++)
	{
		// Convert integer to string
		char path2[150] = "/scratch/abdullsm/GB24_stage0/exageostat_GB/ERA_data/data_";
		sprintf(numStr, "%d", y);
		strcat(path2 , numStr);
		strcat(path2, ".nc");
		ncid    = openFileNCmpi(&data, path2);
		if(isLeapYear(y))
		{	
			t2m = (short int *) malloc((v*24+24) * no_locs * sizeof(int));

			t2m_local = (short int *) malloc((v*24+24) * no_locs * sizeof(int)/nprocs);						
			if(t2m == NULL|| t2m_local == NULL)
				fprintf(stderr, "no enough memory... t2m or t2m_local");
			time_len=v*24+24;
		}
		else
		{	t2m = (short int *) malloc(v * 24 * no_locs* sizeof(int));
			t2m_local = (short int *) malloc((v*24) * no_locs* sizeof(int)/nprocs);
			if(t2m == NULL|| t2m_local == NULL)
				fprintf(stderr, "no enough memory... t2m or t2m_local");
			time_len=v*24;
		}

		double x_read=0;
		START_TIMING(x_read);
		size_t index[] = {(time_len/nprocs)*rank, lts, 0};		
		size_t count[] = {time_len/nprocs, 1, no_locs};

		VERBOSE("Start reading......\n");
		if (retval = ncmpi_get_vara_int_all(ncid, t2m_varid, index, count, t2m_local))
			printf("ncmpi_get_vara_int_all: Error: %s -- %s \n", nc_strerror(retval), path2);	
		if (retval =ncmpi_get_att_double(ncid, t2m_varid, "scale_factor", &scaling_var))
			printf("ncmpi_get_att_double: Error: %s\n", nc_strerror(retval));
		if (retval =ncmpi_get_att_double(ncid, t2m_varid, "add_offset", &offset_var))
			printf("ncmpi_get_att_double: Error: %s\n", nc_strerror(retval));

                VERBOSE("End reading.\n");

		MPI_Allgather(t2m_local, (time_len/nprocs)*no_locs, MPI_INT, t2m, (time_len/nprocs)*no_locs, MPI_INT, MPI_COMM_WORLD);
                VERBOSE("End gathering.\n");
/*
		int sum =0;
		for (int loc_temp=0;loc_temp<100;loc_temp++)
		{	sum = 0;
			for (size_t iter=0;iter<time_len;iter++)
			{
				sum+=t2m[(iter*1440)+loc_temp];
			//	fprintf(stderr, "%d\n", t2m[(iter+loc_temp)+iter*1440]);	
			}

			fprintf(stderr, "loc: %d  sum : %d time_len: %d\n", loc_temp, sum, time_len);
		}

		exit(0);
*/

		STOP_TIMING(x_read);
#if defined(CHAMELEON_USE_MPI)
		if(MORSE_My_Mpi_Rank() == 0)
		{
#endif		
			fprintf(stderr, "================================ \n");	
			fprintf(stderr, "%s\n", path2);
			fprintf(stderr, "time to read the NETCDF file: %f secs\n", x_read);
#if defined(CHAMELEON_USE_MPI)
		}
#endif


		//		for (int x=0;x<20;x++)
		//		{
		//#if defined(CHAMELEON_USE_MPI)
		//			if(MORSE_My_Mpi_Rank() == 0)
		//			{
		//#endif
		//					fprintf(stderr, "%d\n", t2m[x]);
		//#if defined(CHAMELEON_USE_MPI)
		//			}
		//#endif
		//		}
		//exit(0);
		//exit(0);
		//		if( y ==1941){	
		//			int sum_t=0;
		//			for(int k=0;k<time_len*2;k+=2)
		//			{
		//				sum_t+=t2m[k];

		//			}

		//	fprintf(stderr, "sum %d, %d, %d, %d, %d, %d, %d\n", sum_t, count[0], count[1], count[2], index[0], index[1], index[2]);
		//			exit(0);
		//		}
		//******************
		//		int sumi =0;
		//		for(int ii=0;ii<time_len*no_locs ;ii++)
		//		{       
		//			fprintf(stderr, "%d\n", t2m[ii]);
		//			sumi+=t2m[ii];
		//		}
		//		fprintf(stderr," sum=======: %d\n", sumi);
		//		exit(0);
		//******************




		int num_locs=0;	
		for (int lu=0;lu<no_locs; lu++)
		{	
			//fprintf(stderr, ">>> %d %d\n", num_locs, no_locs);			
			double sum_temp = 0;
			int sum_tmp=0;
			//ll=0;
			//to be removed
			// 8784
			int r=0;
			if(isLeapYear(y))
			{
				r=0;
				//fprintf(stderr, ">>> %d %d\n", num_locs, no_locs);
				for(int k=lu;k<(v*24+24)*no_locs;k+=no_locs)
				{

					if(r>= 1416 && r<=1439);
					else
					{	
					//	sum_temp += ((double)t2m[k]*scaling_var) + offset_var - 273.15;
					//	if((r+1) % 24 == 0 )
					//	{	
							t2m_hourly_per_year[lu][t2m_hourly_per_year_count[lu]++]=((double)t2m[k]*scaling_var) + offset_var - 273.15;// = sum_temp/24.0;
if((t2m_hourly_per_year[lu][t2m_hourly_per_year_count[lu]-1]) >10)
fprintf(stderr, " large value : %lf \n", (t2m_hourly_per_year[lu][t2m_hourly_per_year_count[lu]-1]));
					//		sum_temp = 0;
					//	}
					}
					r++;
				}
			}
			else
			{
				r=0;
				for(int k=lu;k<v*24*no_locs;k+=no_locs)
				{
					//sum_temp += ((double)t2m[k]*scaling_var) + offset_var - 273.15;
					//if((r+1) % 24 == 0 )
					//{
						t2m_hourly_per_year[lu][t2m_hourly_per_year_count[lu]++] = ((double)t2m[k]*scaling_var) + offset_var - 273.15; //sum_temp/24.0;
if((t2m_hourly_per_year[lu][t2m_hourly_per_year_count[lu]-1]) >10)
fprintf(stderr, " large value : %lf \n", (t2m_hourly_per_year[lu][t2m_hourly_per_year_count[lu]-1]));
					//	sum_temp = 0;
					//}
					r++;
				}
				num_locs++;
			}
		}
		//*****************************
	
//		free(t2m_local);
//		free(t2m);
		closeFileNCmpi(&data, ncid);//	free (path2);
	}

/*
                double sum =0;
                for (int loc_temp=0;loc_temp<100;loc_temp++)
                {       sum = 0;
                        for (size_t iter=0;iter<365*83;iter++)
                        {
                                sum+= t2m_hourly_per_year[loc_temp][iter];
                        //      fprintf(stderr, "%d\n", t2m[(iter+loc_temp)+iter*1440]);
                        }

                        fprintf(stderr, "loc: %d  sum : %f time_len: %d\n", loc_temp, sum, 730);
                }

                exit(0);
*/

	//	double summ =0;
	//	for(int r=0;r<730;r++)
	//	{
	//		summ+=t2m_hourly_per_year[0][r];
	//#if defined(CHAMELEON_USE_MPI)
	//		if(MORSE_My_Mpi_Rank() == 0)
	//		{
	//#endif
	//			fprintf(stderr, "%f\n", t2m_hourly_per_year[1][r]);
	//#if defined(CHAMELEON_USE_MPI)
	//		}
	//#endif
	//	}
	//	fprintf(stderr, " ---->summ :%f\n", summ);
	//	exit(0);

	//	free( t2m_local);
	//	free( t2m);

//	for(int k=0;k<no_locs;k++)
//	{
//		free(t2m_hourly_per_year[k]);// = (double *) malloc(v * (2023-1988)  *   sizeof(double));
//	}
//	free(t2m_local);
                              //          fprintf(stderr, "hi1\n");
	MORSE_dmle_Call(&data, ncores, gpus, dts, p_grid, q_grid, N,  0, 0);
                            //            fprintf(stderr, "hi2\n");
	//MORSE_dmle_Call(&data, ncores, gpus, dts, p_grid, q_grid, N,  0, 0);
	int count = 0;
	nlopt_set_max_objective(opt, MLE_alg, (void *)&data);
/*
	double summ=0;
	for(int r=0;r<v*(2023-1988);r++)
	{
		summ+=t2m_hourly_per_year[0][r];
	}
	fprintf(stderr, "(before)Sum for this location: %f:  count:%d\n: pointer: %p", summ, 19, &t2m_hourly_per_year[19][0]);
*/
                           //             fprintf(stderr, "hi3\n");

	for (int u=lts;u<lts+1; u++)
		for (int l=0;l<no_locs; l++)//no_locs
		{

			fprintf(stderr, "u=%d, l=%d\n", u, l);
			//if(l ==data.hicma_maxrank)
			if(true)	
			{
				data.iter_count=0;

				//	double sum =0;
				//	for(int k=0;k<365*83;k++)
				//	{
				//		sum+=t2m_hourly_per_year[0][k];
				//	}
				//fprintf(stderr, " sum :%f\n", sum);
				//exit(0);
				//                                        fprintf(stderr, "%d \n", MORSE_My_Mpi_Rank());
#if defined(CHAMELEON_USE_MPI)
				if(MORSE_My_Mpi_Rank() == 0)
				{
#endif
					//		fprintf(stderr, "hi5\n");
					//fprintf(stderr, "hi5\n");

					MORSE_MLE_dzcpy(&data, &t2m_hourly_per_year[count][0]);
					//MORSE_Lapack_to_Tile( t2m_hourly_per_year[count], v*(2023-1988), data.descZ);
					START_TIMING(data.total_exec_time);
					nlopt_set_max_objective(opt, MLE_alg, (void *)&data);
					starting_theta[0] = 0.9;
					nlopt_optimize(opt, starting_theta, &opt_f);
					STOP_TIMING(data.total_exec_time);
					//	starting_theta[0] = 0.272049;
					fprintf(stderr, "final theta: %f\n", starting_theta[0]);

					mean_trend(starting_theta, &data, l, u);
					//			exit(0);	


#if defined(CHAMELEON_USE_MPI)
				}        
#endif
			}
			count++;
			//fprintf(stderr, "%d53- hi");
			//exit(0);


			//	exit(0);
			//data.part2_vector=0;

			//			MORSE_Desc_Destroy( (MORSE_desc_t **)&(data.descC) );
			//			MORSE_Desc_Destroy( (MORSE_desc_t **)&(data.descZ) );

			//			MORSE_Desc_Destroy( (MORSE_desc_t **)&(data.X) );
			//			MORSE_Desc_Destroy( (MORSE_desc_t **)&(data.part2_vector) );
			//			MORSE_Desc_Destroy( (MORSE_desc_t **)&(data.estimated_mean_trend) );
			//			MORSE_Desc_Destroy( (MORSE_desc_t **)&(data.XtX) );

		}
	//fprintf(stderr, "longitude: %d latitude: %d \n", l, u);
	//}

	//Phase(2)
	//	int L = 720;
	//	if(strcmp (data.computation, "exact") == 0)
	//		MORSE_dmle_Call_phase_2(&data, ncores, gpus, dts, p_grid, q_grid, N,  L);

	//	read_csv_files();
	//
	//	phase_2_calcs();


fprintf(stderr, "Done\n");
exit(0);


MORSE_Finalize();
return 0;
}

