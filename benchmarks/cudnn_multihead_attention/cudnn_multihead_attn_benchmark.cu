/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


/* The idea behind Self attention is essentially the idea that a neural net
 * can use some of its internally generated feature maps as weights that its
 * other internally generated feature maps can forward pass through. 
 * 
 * Run with two steps:
 * python multihead_attn_model_setup.py
 * buck run @mode/opt //param_bench/train/compute/cpp/fb/cudnn:cudnn_multihead_attn_benchmark -c fbcode.enable_gpu_sections=true -- --training=false --iterations=100
*/

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>

#include <sqlite3.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <gflags/gflags.h>


#define FatalError(s)                                                 \
  do {                                                                \
    std::stringstream _where, _message;                               \
    _where << __FILE__ << ':' << __LINE__;                            \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__; \
    std::cerr << _message.str() << "\nAborting...\n";                 \
    cudaDeviceReset();                                                \
    exit(1);                                                          \
  } while (0)

#define checkCudnnError(status)                                        \
  do {                                                            \
    std::stringstream _error;                                     \
    if (status != CUDNN_STATUS_SUCCESS) {                         \
      _error << "CUDNN failure: " << cudnnGetErrorString(status); \
      FatalError(_error.str());                                   \
    }                                                             \
  } while (0)

#define checkCudaErrors(status)             \
  do {                                      \
    std::stringstream _error;               \
    if (status != 0) {                      \
      _error << "Cuda failure: " << status; \
      FatalError(_error.str());             \
    }                                       \
  } while (0)

#define AT_CUDNN_CHECK checkCudnnError

cudnnStatus_t status;
cudnnHandle_t cudnnHandle;

struct TestData {
	std::string dataset_id;
	std::unordered_map<std::string, double> args_map;
	std::unordered_map<std::string, std::vector<float>> params_map;
};

TestData get_test_data_from_sqlite(std::string db_name, bool print_mode) {

	TestData config;

	sqlite3 *db;
	sqlite3_stmt *stmt;
    
	std::cout<<"Reading from "<<db_name<<"\n";
	int rc = sqlite3_open(db_name.c_str(), &db);

	sqlite3_exec(db, "pragma journal_mode=wal", 0, 0, 0);

	std::string sql = "SELECT * FROM TestData WHERE dataset_id = (SELECT MAX(dataset_id) from TestData)";
	sqlite3_exec(db, "BEGIN TRANSACTION", NULL, NULL, NULL);
	sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, NULL);
	while (sqlite3_step(stmt) != SQLITE_DONE) {
		std::string key = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
		int blobSize = sqlite3_column_bytes(stmt, 2);	
		if (key != "x" && key != "y_pred" && key != "y_true" && key.find("_p") == std::string::npos)
		{
			double val = sqlite3_column_double(stmt, 2);
			config.args_map[key] = val;
		}
		else {
            config.params_map[key] = std::vector<float>();
			float * blob = (float*)sqlite3_column_blob(stmt, 2);			
			for (int i = 0; i < blobSize / sizeof(float); i++) {
				config.params_map[key].push_back(blob[i]);
				if (print_mode)
					std::cout<<"config.params_map["<<key<<"]["<<i<<"]="<<blob[i]<<"\n";
			}      
		}
		if (config.dataset_id.empty())
			config.dataset_id = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
	}
	sqlite3_finalize(stmt);
	sqlite3_exec(db, "END TRANSACTION", NULL, NULL, NULL);
	sqlite3_close(db);
	return config;
};

void store_output_in_sqlite3_db(void * output_d, TestData config) {
	auto s = config.params_map["input_seq_data"];
	size_t output_size = s.size()*sizeof(float);
	void * output_h = malloc(output_size);
	checkCudaErrors(cudaMemcpy(output_h, output_d, output_size, cudaMemcpyDeviceToHost));
 	sqlite3 * db;
	sqlite3_stmt *stmt;
 	int rc = sqlite3_open("attention_net.db", &db);
 	sqlite3_exec(db, "pragma journal_mode=wal", 0, 0, 0);
	const char * sql = "INSERT INTO TestData(test_time, name, data) values (?1, output, ?2)";
	sqlite3_exec(db, "BEGIN TRANSACTION", NULL, NULL, NULL);
	sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);	
	sqlite3_bind_text(stmt, 1, config.dataset_id.c_str(), -1, SQLITE_STATIC);
	sqlite3_bind_blob(stmt, 2, output_h, output_size, SQLITE_STATIC);
	sqlite3_step(stmt);
	sqlite3_reset(stmt);
	sqlite3_finalize(stmt);
	sqlite3_exec(db, "END TRANSACTION", NULL, NULL, NULL);
	sqlite3_close(db);	
}

DEFINE_bool(debug_mode, false, "Print model data, inputs, and outputs");
DEFINE_bool(training, true, "Specifies whether running inference only or training mode.");
DEFINE_int32(iterations, 100, "Specifies # of forward and/or backward passes.");
DEFINE_string(modelconfig, "./multihead_attn_model_data.db", "Specifies full path to multihead attention model & config data db.");

int main(int argc, char** argv) {

	checkCudnnError(cudnnAdvInferVersionCheck());

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	bool print_mode = FLAGS_debug_mode;
	bool training = FLAGS_training;	

	std::filesystem::path db_name = FLAGS_modelconfig;	
	if (!std::filesystem::exists(db_name)) {
		std::cout<<"\n\n"<<db_name.string()<<" does not exists.\n";
		std::cout<<"Run python multihead_attn_model.py to create the db file containing the test dataset and config.\n\n";
		return 0;
	}
	TestData config = get_test_data_from_sqlite(db_name.string(), print_mode);

	int batch_size 		=  training ? 16 : 16;
	int emb_dim 		=  config.args_map["emb_dim"];
  	const int num_heads =  config.args_map["num_heads"];
	int seq_len 		=  config.args_map["seq_len"];
	int beam_dim 		=  1;
	int seqLengthVecSize = batch_size;
	std::vector<int> seqLengthVec;
	for (int i = 0; i < seqLengthVecSize; i++) 
		seqLengthVec.push_back(seq_len);
		
	cudnnAttnDescriptor_t attn_desc;
	cudnnSeqDataDescriptor_t q_desc = NULL;
	cudnnSeqDataDescriptor_t k_desc = NULL; 
	cudnnSeqDataDescriptor_t v_desc = NULL; 
	cudnnSeqDataDescriptor_t o_desc = NULL;
	
	// The dropout option is currently not supported by the multi-head attention API. 
	cudnnDropoutDescriptor_t attnDropoutDesc = NULL;
	cudnnDropoutDescriptor_t postDropoutDesc = NULL;	

	unsigned int attnMode = CUDNN_ATTN_DISABLE_PROJ_BIASES;
	double smScaler = 1.0;
	cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
	cudnnDataType_t computePrec = CUDNN_DATA_FLOAT;
	cudnnMathType_t mathType = CUDNN_DEFAULT_MATH;

	int qSize = emb_dim;
	int kSize = emb_dim;
	int vSize = emb_dim;
	int qProjSize = emb_dim / num_heads;
	int kProjSize = emb_dim / num_heads;
	int vProjSize = emb_dim / num_heads;
	int oProjSize = emb_dim;
	int qoMaxSeqLength = seq_len;
	int kvMaxSeqLength = seq_len;
	int maxBatchSize = batch_size;
	int maxBeamSize = 1;

	size_t sizeWeights = 0;
	size_t sizeWkspace = 0;
	size_t sizeReserve = 0;

	void * devW = nullptr;
	void * ddevW = nullptr;
	void * devWkspace = nullptr;
	void * devReserve = nullptr;
	void * queries_w_d = nullptr;
	void * devDQ = nullptr;	

	cudnnTensorDescriptor_t queries_w_desc;
	cudnnTensorDescriptor_t keys_w_desc;
	cudnnTensorDescriptor_t values_w_desc;
	cudnnTensorDescriptor_t out_w_desc;
	void * keys_w_d = nullptr;
	void * devDK = nullptr;
	void * values_w_d = nullptr;
	void * devDV = nullptr;	
	void * out_w_d = nullptr;
	void * devDO = nullptr;	
	void * devQ = nullptr;
	void * devK = nullptr;
	void * devV = nullptr;
	void * y_pred_d = nullptr;

	/* " Device array specifying sequence lengths of query, residual, and output sequence data." */
	int * devQSeqArray;
	/* "Device array specifying sequence lengths of key and value input data." */
	int * devKSeqArray;
 	/* "host arrays specify the attention window size for each Q time-step." */
	std::vector<int> loWinIdx, hiWinIdx;		

	cudnnSeqDataAxis_t axes[CUDNN_SEQDATA_DIM_COUNT];
	axes[3] = CUDNN_SEQDATA_VECT_DIM;   
	axes[0] = CUDNN_SEQDATA_BEAM_DIM;
	axes[2] = CUDNN_SEQDATA_BATCH_DIM;
	axes[1] = CUDNN_SEQDATA_TIME_DIM;

	int dimA[CUDNN_SEQDATA_DIM_COUNT];
	dimA[CUDNN_SEQDATA_TIME_DIM]  = seq_len;
	dimA[CUDNN_SEQDATA_BATCH_DIM] = batch_size;
	dimA[CUDNN_SEQDATA_BEAM_DIM]  = beam_dim;
	dimA[CUDNN_SEQDATA_VECT_DIM]  = emb_dim;

	checkCudnnError(cudnnCreate(&cudnnHandle));

	checkCudnnError(cudnnCreateTensorDescriptor(&queries_w_desc));
	checkCudnnError(cudnnCreateTensorDescriptor(&keys_w_desc));
	checkCudnnError(cudnnCreateTensorDescriptor(&values_w_desc));
	checkCudnnError(cudnnCreateTensorDescriptor(&out_w_desc));	

	checkCudnnError(cudnnCreateAttnDescriptor(&attn_desc));

	checkCudnnError(cudnnCreateSeqDataDescriptor(&q_desc));
	checkCudnnError(cudnnCreateSeqDataDescriptor(&k_desc));
	checkCudnnError(cudnnCreateSeqDataDescriptor(&v_desc));
	checkCudnnError(cudnnCreateSeqDataDescriptor(&o_desc));

	checkCudnnError(cudnnSetAttnDescriptor(
			attn_desc,
			attnMode,
			num_heads,		
			smScaler,
			dataType,
			computePrec,
			mathType,
			attnDropoutDesc,
			postDropoutDesc,
			qSize,
			kSize,
			vSize,
			qProjSize,
			kProjSize,
			vProjSize,
			oProjSize,
			qoMaxSeqLength,
			kvMaxSeqLength,
			maxBatchSize,
			maxBeamSize
		)
	);
	
	checkCudnnError(cudnnGetMultiHeadAttnBuffers(
			cudnnHandle,
			attn_desc,
			&sizeWeights,
			&sizeWkspace,
			training ? &sizeReserve : NULL
	));	

	checkCudaErrors(cudaMalloc((void**)&devW, sizeWeights));
	checkCudaErrors(cudaMalloc((void**)&devWkspace, sizeWkspace));
	if (training) {
		checkCudaErrors(cudaMalloc((void**)&ddevW, sizeWeights));	
		checkCudaErrors(cudaMalloc((void**)&devReserve, sizeReserve));
	}

	auto param = config.params_map["q_p.weight"];
	checkCudnnError(cudnnGetMultiHeadAttnWeights(
			cudnnHandle,
			attn_desc,
			CUDNN_MH_ATTN_Q_WEIGHTS,
			sizeWeights,
			devW,
			queries_w_desc,
			&queries_w_d));	
	checkCudaErrors(cudaMemcpy(queries_w_d, param.data(), sizeof(float) * param.size(), cudaMemcpyHostToDevice));
	if (training)
		checkCudaErrors(cudaMalloc((void**)&devDQ, sizeof(float) * param.size()));

	param = config.params_map["k_p.weight"];
	checkCudnnError(cudnnGetMultiHeadAttnWeights(
			cudnnHandle,
			attn_desc,
			CUDNN_MH_ATTN_K_WEIGHTS,
			sizeWeights,
			devW,
			keys_w_desc,
			&keys_w_d));	
	checkCudaErrors(cudaMemcpy(keys_w_d, param.data(), sizeof(float) * param.size(), cudaMemcpyHostToDevice));
	if (training)
		checkCudaErrors(cudaMalloc((void**)&devDK, sizeof(float) * param.size()));

	param = config.params_map["v_p.weight"];
	checkCudnnError(cudnnGetMultiHeadAttnWeights(
			cudnnHandle,
			attn_desc,
			CUDNN_MH_ATTN_V_WEIGHTS,
			sizeWeights,
			devW,
			values_w_desc,
			&values_w_d));	
	checkCudaErrors(cudaMemcpy(values_w_d, param.data(), sizeof(float) * param.size(), cudaMemcpyHostToDevice));	
	if (training)
		checkCudaErrors(cudaMalloc((void**)&devDV, sizeof(float) * param.size()));

	param = config.params_map["o_p.weight"];
	checkCudnnError(cudnnGetMultiHeadAttnWeights(
			cudnnHandle,
			attn_desc,
			CUDNN_MH_ATTN_O_WEIGHTS,
			sizeWeights,
			devW,
			out_w_desc,
			&out_w_d));
	checkCudaErrors(cudaMemcpy(out_w_d, param.data(), sizeof(float) * param.size(), cudaMemcpyHostToDevice));
	if (training)
		checkCudaErrors(cudaMalloc((void**)&devDO, sizeof(float) * param.size()));

	param = config.params_map["x"];
	checkCudaErrors(cudaMalloc((void**)&devQ, sizeof(float) * param.size()));
	checkCudaErrors(cudaMalloc((void**)&devK, sizeof(float) * param.size()));
	checkCudaErrors(cudaMalloc((void**)&devV, sizeof(float) * param.size()));
	checkCudaErrors(cudaMemcpy(devQ, param.data(), sizeof(float) * param.size(), cudaMemcpyHostToDevice));	
	checkCudaErrors(cudaMemcpy(devK, param.data(), sizeof(float) * param.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(devV, param.data(), sizeof(float) * param.size(), cudaMemcpyHostToDevice));

	param = config.params_map["y_pred"];
	checkCudaErrors(cudaMalloc((void**)&y_pred_d, sizeof(float) * param.size()));

	checkCudnnError(cudnnSetSeqDataDescriptor(
			q_desc, 
			CUDNN_DATA_FLOAT, 
			CUDNN_SEQDATA_DIM_COUNT, 
			dimA, 
			axes,
			seqLengthVecSize,
			seqLengthVec.data(),
			NULL
	));
	checkCudnnError(cudnnSetSeqDataDescriptor(
			k_desc, 
			CUDNN_DATA_FLOAT, 
			CUDNN_SEQDATA_DIM_COUNT, 
			dimA, 
			axes,
			seqLengthVecSize,
			seqLengthVec.data(),
			NULL
	));
	checkCudnnError(cudnnSetSeqDataDescriptor(
			v_desc, 
			CUDNN_DATA_FLOAT, 
			CUDNN_SEQDATA_DIM_COUNT, 
			dimA, 
			axes,
			seqLengthVecSize,
			seqLengthVec.data(),
			NULL
	));
	checkCudnnError(cudnnSetSeqDataDescriptor(
			o_desc, 
			CUDNN_DATA_FLOAT, 
			CUDNN_SEQDATA_DIM_COUNT, 
			dimA, 
			axes,
			seqLengthVecSize,
			seqLengthVec.data(),
			NULL
	));		

	std::vector<int>  devQSeqArray_vec(batch_size * beam_dim, seq_len);
	std::vector<int>  devKSeqArray_vec(batch_size, seq_len);

	int devQSeqArray_size = devQSeqArray_vec.size() * sizeof(int);
	int devKSeqArray_size = devKSeqArray_vec.size() * sizeof(int);

	checkCudaErrors(cudaMalloc((void**)&devQSeqArray, devQSeqArray_size));	
	checkCudaErrors(cudaMalloc((void**)&devKSeqArray, devKSeqArray_size));

	checkCudaErrors(cudaMemcpy(devQSeqArray, (void*)devQSeqArray_vec.data(), devQSeqArray_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(devKSeqArray, (void*)devKSeqArray_vec.data(), devKSeqArray_size, cudaMemcpyHostToDevice));

	int maxSeqLenK = INT_MAX;	
	for (int i = 0; i < seq_len; i++) {
		loWinIdx.push_back(0);
		hiWinIdx.push_back(maxSeqLenK);
	}
	checkCudaErrors(cudaDeviceSynchronize());
	auto t1 = std::chrono::high_resolution_clock::now();	
	for (int iter = 0; iter < FLAGS_iterations; ++iter) {
		int currIdx=-1;												
		checkCudnnError(cudnnMultiHeadAttnForward(  // parameter names in CUDNN API docs 
				cudnnHandle, 						// cudnnHandle_t handle
				attn_desc,							// const cudnnAttnDescriptor_t attn_desc	
				currIdx, 							// int currIdx
				(const int*)loWinIdx.data(),		// const int loWinIdx[]
				(const int*)hiWinIdx.data(),		// const int hiWinIdx[]
				devQSeqArray,						// const int devQSeqArray[],
				devKSeqArray, 						// const int devKSeqArray[],
				q_desc,								// const cudnnSeqDataDescriptor_t q_desc
				devQ,								// const void *queries,
				NULL,								// const void *residuals,
				k_desc,								// const cudnnSeqDataDescriptor_t k_desc,
				devK,								// const void *keys,
				v_desc,								// const cudnnSeqDataDescriptor_t v_desc,
				devV,								// const void *values,
				o_desc,								// const cudnnSeqDataDescriptor_t o_desc,
				y_pred_d,							// void *out
				sizeWeights,						// size_t weightSizeInBytes,
				devW,								// const void *weights,
				sizeWkspace,						// size_t workSpaceSizeInBytes,
				devWkspace,							// void *workSpace,
				sizeReserve,						// size_t reserveSpaceSizeInBytes,
				devReserve							// void *reserveSpace		
		));
		if (training) {		
			checkCudnnError(cudnnMultiHeadAttnBackwardData(
					cudnnHandle,
					attn_desc,	 	
					(const int*)loWinIdx.data(),	
					(const int*)hiWinIdx.data(),	
					devQSeqArray,
					devKSeqArray, 
					o_desc,		
					devDO,		
					q_desc,
					devDQ,		
					devQ,		
					k_desc,
					devDK,
					devK,		
					v_desc,
					devDV,
					devV,			
					sizeWeights,
					devW,
					sizeWkspace,
					devWkspace,
					sizeReserve,
					devReserve));			
			checkCudnnError(cudnnMultiHeadAttnBackwardWeights(
					cudnnHandle,
					attn_desc,
					CUDNN_WGRAD_MODE_SET,	
					q_desc,
					devQ,		
					k_desc,
					devK,		
					v_desc,
					devV,	
					o_desc,
					devDO,
					sizeWeights,
					devW,
					ddevW,
					sizeWkspace,
					devWkspace,
					sizeReserve,
					devReserve
			));
		}				
	}

	checkCudaErrors(cudaDeviceSynchronize());
	auto t2 = std::chrono::high_resolution_clock::now();
	double dur_time =
		std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() /
		1000.0f / FLAGS_iterations;
    printf("Iteration time: %lf ms\n", dur_time);
	printf(
		"TF/s: %lf\n",
		2.0 * batch_size * 
		config.args_map["params_count"]
		/dur_time / 1000000000.0f);	

	if (print_mode) {
		float * weights_buffer_h = (float*)malloc(sizeWeights);
		cudaMemcpy(weights_buffer_h, devW, sizeWeights, cudaMemcpyDeviceToHost);
		for (int i = 0; i < sizeWeights/sizeof(float); i++) {
			std::cout<<"weights_buffer_h["<<i<<"]="<<weights_buffer_h[i]<<"\t\t";
		}
		std::cout<<"\n\n";

		std::cout<<"x (nn input):\n";
		for (int i = 0; i < 100 && i < config.params_map["x"].size(); i++)
			std::cout<<config.params_map["x"][i]<<"\n";

		int size  = config.params_map["y_pred"].size();
		float * y_pred_h = (float*)malloc(size*sizeof(float));	
		checkCudaErrors(cudaMemcpy(y_pred_h, y_pred_d, size*sizeof(float), cudaMemcpyDeviceToHost));
		std::cout<<"\nconfig.params_map[y_pred].size(): "<<config.params_map["y_pred"].size()<<std::endl;
		std::cout<<"size: "<<size<<std::endl;
		for (int i = 0; i < 100 && i < size; i++)
			std::cout<<"CuDNN y_pred: "<<y_pred_h[i]<<"\t\t"<<"PyTorch y_pred: "<<config.params_map["y_pred"][i]<<"\n";
		free(y_pred_h);
	}
	cudaFree(devW);
	cudaFree(ddevW);
	cudaFree(devWkspace);
	cudaFree(devQSeqArray);
	cudaFree(devKSeqArray);
	cudaFree(y_pred_d);	
	cudaFree(devDO);
	return 0;
} 
