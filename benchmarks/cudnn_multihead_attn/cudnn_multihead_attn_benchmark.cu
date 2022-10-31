/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <sqlite3.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <gflags/gflags.h>

DEFINE_bool(double_precision, true, "Whether to use double precision.");
DEFINE_bool(debug_mode, false, "Print model tensors, inputs, and outputs");
DEFINE_int32(print_limit_per_tensor, 10, "Specifies # of elements to print per tensor when running in debug mode.");
DEFINE_bool(print_accuracy_stats, false, "Print CuDNN's accuracy compared to the reference data.");
DEFINE_bool(training, true, "Whether to run inference only or training mode.");
DEFINE_int32(iterations, 100, "Specifies # of forward and/or backward passes.");
DEFINE_string(ref_db, "./multihead_attn_ref.db", "Specifies the full path to the multi-head attention reference file.");

#define FatalError(s)                                                 \
  do {                                                                \
    std::stringstream _where, _message;                               \
    _where << __FILE__ << ':' << __LINE__;                            \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__; \
    std::cerr << _message.str() << "\nAborting...\n";                 \
    cudaDeviceReset();                                                \
    exit(1);                                                          \
  } while (0)

#define checkCudnnError(status)                                   \
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

cudnnStatus_t status;
cudnnHandle_t cudnnHandle;

template<typename T>
struct RefData {
	std::string dataset_id;
	std::unordered_map<std::string, double> args;
	std::unordered_map<std::string, std::vector<T>> vecs;
};

template<typename T>
RefData<T> get_ref_data_from_sqlite(std::string db_name, bool debug_mode) {

	RefData<T> cfg;

	sqlite3 *db;
	sqlite3_stmt *stmt;
    
	std::cout << "\nReading PyTorch reference data from " << db_name << "\n";
	int rc = sqlite3_open(db_name.c_str(), &db);

	sqlite3_exec(db, "pragma journal_mode=wal", 0, 0, 0);

	std::string sql = "SELECT * FROM RefData WHERE dataset_id = (SELECT MAX(dataset_id) from RefData)";
	sqlite3_exec(db, "BEGIN TRANSACTION", NULL, NULL, NULL);
	sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, NULL);
	while (sqlite3_step(stmt) != SQLITE_DONE) {
		std::string key = std::string(
			reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
		bool isTensor = key == "q" || key == "k" || key == "v" || key == "o" 
			|| key.find("_p.") != std::string::npos; 
		if (isTensor) {
			// Load tensor.
			if (debug_mode)
				std::cout<<"Loading tensor "<<key<<"\n";
			cfg.vecs[key] = std::vector<T>();
			if (cfg.args["double_precision_ref_data"] == 1.0) {
				double * blob = (double*)sqlite3_column_blob(stmt, 2);
				int count = sqlite3_column_bytes(stmt, 2) / sizeof(double);
				for (int i = 0; i < count; i++)
					cfg.vecs[key].push_back((T)blob[i]);
			} else {
				float * blob = (float*)sqlite3_column_blob(stmt, 2);
				int count = sqlite3_column_bytes(stmt, 2) / sizeof(float);
				for (int i = 0; i < count; i++)
					cfg.vecs[key].push_back((T)blob[i]);
			}
		} else {
			// Load single value argument.
			double val = sqlite3_column_double(stmt, 2);
			cfg.args[key] = val;
			std::cout << key << "=" << val << "\n";
		}
		if (cfg.dataset_id.empty())
			cfg.dataset_id = std::string(
				reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
	}
	sqlite3_finalize(stmt);
	sqlite3_exec(db, "END TRANSACTION", NULL, NULL, NULL);
	sqlite3_close(db);

	if (debug_mode && false ) {
		std::cout << "\nPrinting up to the first 10 elements of each tensor:\n\n";
		for (std::pair<std::string, std::vector<T>> keyval : cfg.vecs) {
			auto key = keyval.first;
			auto vec = keyval.second;
			for (size_t i = 0; i < vec.size() && i < 10; i++)
				std::cout << key << " [" << i << "] = " << vec[i] << "\n";
			std::cout << " \n";
		}
		std::cout << "Reading from db Done.\n\n";
	}
	return cfg;
};

template<typename T>
int run() {
	checkCudnnError(cudnnAdvInferVersionCheck());

	bool debug_mode = FLAGS_debug_mode;
	bool training = FLAGS_training;	

	std::filesystem::path db_name = FLAGS_ref_db;
	if (std::filesystem::exists(db_name))
		;
	else if (std::filesystem::exists("./cudnn_multihead_attn/multihead_attn_ref.db"))
		db_name = "./cudnn_multihead_attn/multihead_attn_ref.db";
	else {
		std::cout << "\n\nChecking " << db_name.string() << ". File does not exists.\n";
		db_name = "./cudnn_multihead_attn/multihead_attn_ref.db";
		std::cout << "\nNext, checking " << db_name.string() << " File does not exists.\n";
		std::cout
			<< "Run python cudnn_multihead_attn/multihead_attn_make_ref.py to create a reference db file.\n\n";
		return 0;
	}
	RefData cfg = get_ref_data_from_sqlite<T>(db_name.string(), debug_mode);

	int batch_size 		=  cfg.args["batch_size"];
	int emb_dim 		=  cfg.args["emb_dim"];
	const int num_heads =  cfg.args["num_heads"];
	int seq_len 		=  cfg.args["seq_len"];
	int beam_dim 		=  1;
	int seqLensVecSize = batch_size;
	std::vector<int> seqLensVec;
	for (int i = 0; i < seqLensVecSize; i++) 
		seqLensVec.push_back(seq_len);
		
	// q, k, v embedding vector lengths
	int qSize = emb_dim;
	int kSize = emb_dim;
	int vSize = emb_dim;

	// q, k, v embedding vector lengths after input projections. 
	int qProjSize = emb_dim / num_heads;
	int kProjSize = emb_dim / num_heads;
	int vProjSize = emb_dim / num_heads;

	// o embedding vector length after the output projection
	int oProjSize = emb_dim;

	int qoMaxSeqLength = seq_len;
	int kvMaxSeqLength = seq_len;
	int maxBatchSize = batch_size;
	int maxBeamSize = 1;
	
	// ***********************************************************************
	// Variables' Roles:
	//
	// devAQ -> [ devWQ + devBQ ] -> ..
	// Input      Linear Layer          \			         
	//                                   |
	// devAK -> [ devWK + devBK ] -> [ hidden ] -> [ devWO + devBO ] -> devAO
	// Input      Linear Layer           |           Linear Layer       Output
	//                                  /
	// devAV -> [ devWV + devBV ] -> ..
	// Input      Linear Layer
	//
	// ***********************************************************************
			
	// Below variables are used as shown above.
	void * devAQ = nullptr; // q Activations
	void * devAK = nullptr; // k Activations
	void * devAV = nullptr; // v Activations
	void * devAO = nullptr; // o Activations
	void * devWQ = nullptr; // q Linear Layer Weights
	void * devWK = nullptr; // k Linear Layer Weights
	void * devWV = nullptr; // v Linear Layer Weights
	void * devWO = nullptr; // o Linear Layer Weights
	void * devBQ = nullptr; // q Linear Layer Biases
	void * devBK = nullptr; // k Linear Layer Biases
	void * devBV = nullptr; // v Linear Layer Biases
	void * devBO = nullptr; // o Linear Layer Biases	

	// Corresponding partial derivatives.
	void * devDAQ = nullptr;	
	void * devDAK = nullptr;
	void * devDAV = nullptr;	
	void * devDAO = nullptr;		
	void * devDWQ = nullptr;
	void * devDWK = nullptr;
	void * devDWV = nullptr;
	void * devDWO = nullptr;
	void * devDBQ = nullptr;
	void * devDBK = nullptr;
	void * devDBV = nullptr;
	void * devDBO = nullptr;	

	size_t sizeWeights = 0;
	size_t sizeWkspace = 0;
	size_t sizeReserve = 0;
	void * devWs = nullptr;
	void * devDWs = nullptr;
	void * devWkspace = nullptr;
	void * devReserve = nullptr;	

	// Device array specifying seq. lengths of query, residual, and output seq. data.
	int * devQSeqArray = nullptr;
	// Device array specifying seq. lengths of key and value input data.
	int * devKSeqArray = nullptr;
 	// Host arrays specifying the attention window size for each Q time-step.
	std::vector<int> loWinIdx, hiWinIdx;		

	cudnnSeqDataAxis_t axes[CUDNN_SEQDATA_DIM_COUNT];
	axes[0] = CUDNN_SEQDATA_BEAM_DIM;
	axes[1] = CUDNN_SEQDATA_TIME_DIM;
	axes[2] = CUDNN_SEQDATA_BATCH_DIM;
	axes[3] = CUDNN_SEQDATA_VECT_DIM;   

	int dimA[CUDNN_SEQDATA_DIM_COUNT];
	dimA[CUDNN_SEQDATA_BEAM_DIM]  = beam_dim;
	dimA[CUDNN_SEQDATA_TIME_DIM]  = seq_len;
	dimA[CUDNN_SEQDATA_BATCH_DIM] = batch_size;
	dimA[CUDNN_SEQDATA_VECT_DIM]  = emb_dim;
	
	cudnnAttnDescriptor_t attn_desc;
	cudnnSeqDataDescriptor_t q_desc = NULL;
	cudnnSeqDataDescriptor_t k_desc = NULL; 
	cudnnSeqDataDescriptor_t v_desc = NULL; 
	cudnnSeqDataDescriptor_t o_desc = NULL;
	
	// Dropout is currently not supported by CuDNN's multi-head attention API. 
	cudnnDropoutDescriptor_t attnDropoutDesc = NULL;
	cudnnDropoutDescriptor_t postDropoutDesc = NULL;	

	bool enable_bias = !(bool)cfg.args["disable_bias"];
	unsigned int attnMode = 
		enable_bias ? CUDNN_ATTN_ENABLE_PROJ_BIASES : CUDNN_ATTN_DISABLE_PROJ_BIASES;
	double smScaler = 1.0;
	cudnnDataType_t dataType = 
		(sizeof(T) == 8) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
	cudnnDataType_t computePrec = 
		(sizeof(T) == 8) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
	cudnnMathType_t mathType = CUDNN_DEFAULT_MATH;

	checkCudnnError(cudnnCreateAttnDescriptor(&attn_desc));
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
			maxBeamSize));
	checkCudnnError(cudnnCreate(&cudnnHandle));
	checkCudnnError(cudnnGetMultiHeadAttnBuffers(
		cudnnHandle, attn_desc, &sizeWeights, &sizeWkspace, training ? &sizeReserve : NULL));	
	checkCudaErrors(cudaMalloc((void**)&devWs, sizeWeights));
	checkCudaErrors(cudaMalloc((void**)&devWkspace, sizeWkspace));
	if (training) {
		checkCudaErrors(cudaMalloc((void**)&devDWs, sizeWeights));	
		checkCudaErrors(cudaMalloc((void**)&devReserve, sizeReserve));
		checkCudaErrors(cudaMemset(devDWs, 0.0, sizeWeights));
		checkCudaErrors(cudaMemset(devReserve, 0.0, sizeReserve));

		checkCudaErrors(cudaMemset(devWs, 0.0, sizeWeights));
		checkCudaErrors(cudaMemset(devWkspace, 0.0, sizeWkspace));
	}

	struct subblock{ 
		std::string name; 
		void ** devA; 	// used for q, k, v, o activations.
		void ** devW;  	// used for q, k, v, o linear layer weights.
		void ** devB;  	// used for q, k, v, o linear layer biases.
		void ** devDA; 	// used for partial derivatives of q, k, v, o activations.
		void ** devDW;  // used for partial derivatives of q, k, v, o linear layer weights.
		void ** devDB;  // used for partial derivatives of q, k, v, o linear layer biases.
		cudnnMultiHeadAttnWeightKind_t enumW;
		cudnnMultiHeadAttnWeightKind_t enumB;
	};
	
	// Shorten enum names to fit the subblocks table below.
	auto enumWQ = CUDNN_MH_ATTN_Q_WEIGHTS, enumBQ = CUDNN_MH_ATTN_Q_BIASES;
	auto enumWK = CUDNN_MH_ATTN_K_WEIGHTS, enumBK = CUDNN_MH_ATTN_K_BIASES;
	auto enumWV = CUDNN_MH_ATTN_V_WEIGHTS, enumBV = CUDNN_MH_ATTN_V_BIASES;
	auto enumWO = CUDNN_MH_ATTN_O_WEIGHTS, enumBO = CUDNN_MH_ATTN_O_BIASES;
	
	std::vector<subblock> subblocks{
		//
		//  Corresponding struct member names:		
		// .name | .devA | .devW | .devB | .devDA | .devDW | .devDB |.enumW |.enumB
		//
		{"q",     &devAQ, &devWQ, &devBQ, &devDAQ, &devDWQ, &devDBQ, enumWQ, enumBQ},
		{"k",     &devAK, &devWK, &devBK, &devDAK, &devDWK, &devDBK, enumWK, enumBK},
		{"v",     &devAV, &devWV, &devBV, &devDAV, &devDWV, &devDBV, enumWV, enumBV},
		{"o",     &devAO, &devWO, &devBO, &devDAO, &devDWO, &devDBO, enumWO, enumBO},
	};
	for (auto & s : subblocks) {

		auto avec = cfg.vecs[s.name]; 
		auto wvec = cfg.vecs[s.name + "_p.weight"];
		auto bvec = cfg.vecs[s.name + "_p.bias"];

		cudnnTensorDescriptor_t desc;
		checkCudnnError(cudnnCreateTensorDescriptor(&desc));

		// Allocate memory for activations devAQ, devAK, devAV, devAO.
		checkCudaErrors(cudaMalloc(s.devA, sizeof(T) * avec.size()));

		// Store addresses for weights in devWQ, devWK, devWV, devWO.
		checkCudnnError(cudnnGetMultiHeadAttnWeights(
			cudnnHandle, attn_desc, s.enumW, sizeWeights, devWs,  desc, s.devW));

		// Store addresses for biases in devBQ, devBK, devBV, devBO.
		checkCudnnError(cudnnGetMultiHeadAttnWeights(
			cudnnHandle, attn_desc, s.enumB, sizeWeights, devWs,  desc, s.devB));

		if (training) {
			// Allocate memory for activations' gradients devDAQ, devDAK, devDAV, devDAO.
			checkCudaErrors(cudaMalloc(s.devDA, sizeof(T) * avec.size()));

			// Store addresses for weights' gradients in devDWQ, devDWK, devDWV, devDWO.
			checkCudnnError(cudnnGetMultiHeadAttnWeights(
				cudnnHandle, attn_desc, s.enumW, sizeWeights, devDWs, desc, s.devDW));

			// Store addresses for biases' gradients in devDBQ, devDBK, devDBV, devDBO.
			checkCudnnError(cudnnGetMultiHeadAttnWeights(
				cudnnHandle, attn_desc, s.enumB, sizeWeights, devDWs, desc, s.devDB));
		}
		// Copy PyTorch reference weights to GPU. 
		checkCudaErrors(cudaMemcpy(
			*s.devW, wvec.data(), sizeof(T) * wvec.size(), cudaMemcpyHostToDevice));

		// Copy PyTorch reference biases to GPU. 
		checkCudaErrors(cudaMemcpy(
			*s.devB, bvec.data(), sizeof(T) * bvec.size(), cudaMemcpyHostToDevice));			

		// Copy PyTorch reference inputs to GPU.
		if (s.name == "q" || s.name == "k" || s.name == "v")
			checkCudaErrors(cudaMemcpy(
				*s.devA, avec.data(), sizeof(T) * avec.size(), cudaMemcpyHostToDevice));
	}
	if (training) {
		// Copy gradients that will propagate backward through the entire net, to GPU.
		std::vector<T> DAO (cfg.vecs["o"].size(), 1.0);
		checkCudaErrors(cudaMemcpy(
			devDAO, DAO.data(), sizeof(T) * DAO.size(), cudaMemcpyHostToDevice));	
	}
	for (cudnnSeqDataDescriptor_t * desc: { &q_desc, &k_desc, &v_desc, &o_desc }) {
		checkCudnnError(cudnnCreateSeqDataDescriptor(desc));
		checkCudnnError(cudnnSetSeqDataDescriptor(
			*desc, dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, axes, seqLensVecSize, seqLensVec.data(), NULL));
	}	

	std::vector<int>  hostQSeqVec(batch_size * beam_dim, seq_len);
	std::vector<int>  hostKSeqVec(batch_size, seq_len);
	int qSeqArraySize = hostQSeqVec.size() * sizeof(int);
	int kSeqArraySize = hostKSeqVec.size() * sizeof(int);
	
	checkCudaErrors(cudaMalloc((void**)&devQSeqArray, qSeqArraySize));	
	checkCudaErrors(cudaMalloc((void**)&devKSeqArray, kSeqArraySize));
	checkCudaErrors(cudaMemcpy(
		devQSeqArray, (void*)hostQSeqVec.data(), qSeqArraySize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(
		devKSeqArray, (void*)hostKSeqVec.data(), kSeqArraySize, cudaMemcpyHostToDevice));

	int maxSeqLenK = INT_MAX;	
	for (int i = 0; i < seq_len; i++) {
		loWinIdx.push_back(0);
		hiWinIdx.push_back(maxSeqLenK);
	}
	checkCudaErrors(cudaDeviceSynchronize());

	auto t1 = std::chrono::high_resolution_clock::now();
	for (int iter = 0; iter < FLAGS_iterations; ++iter) {
		int currIdx=-1;												
		checkCudnnError(cudnnMultiHeadAttnForward(
				                // parameter names in CuDNN API docs 
				cudnnHandle,    // cudnnHandle_t handle
				attn_desc,      // const cudnnAttnDescriptor_t attn_desc	
				currIdx,        // int currIdx
				(const int*)loWinIdx.data(),    // const int loWinIdx[]
				(const int*)hiWinIdx.data(),    // const int hiWinIdx[]
				devQSeqArray,   // const int devQSeqArray[],
				devKSeqArray,   // const int devKSeqArray[],
				q_desc,         // const cudnnSeqDataDescriptor_t q_desc
				devAQ,          // const void *queries,
				NULL,           // const void *residuals,
				k_desc,         // const cudnnSeqDataDescriptor_t k_desc,
				devAK,          // const void *keys,
				v_desc,         // const cudnnSeqDataDescriptor_t v_desc,
				devAV,          // const void *values,
				o_desc,         // const cudnnSeqDataDescriptor_t o_desc,
				devAO,          // void *out
				sizeWeights,    // size_t weightSizeInBytes,
				devWs,          // const void *weights,
				sizeWkspace,    // size_t workSpaceSizeInBytes,
				devWkspace,     // void *workSpace,
				sizeReserve,    // size_t reserveSpaceSizeInBytes,
				devReserve));   // void *reserveSpace		
		if (training) {		
			checkCudnnError(cudnnMultiHeadAttnBackwardData(
					cudnnHandle,
					attn_desc,	 	
					(const int*)loWinIdx.data(),	
					(const int*)hiWinIdx.data(),	
					devQSeqArray,
					devKSeqArray, 
					o_desc,		
					devDAO,		
					q_desc,
					devDAQ,		
					devAQ,		
					k_desc,
					devDAK,
					devAK,		
					v_desc,
					devDAV,
					devAV,			
					sizeWeights,
					devWs,
					sizeWkspace,
					devWkspace,
					sizeReserve,
					devReserve));			
			checkCudnnError(cudnnMultiHeadAttnBackwardWeights(
					cudnnHandle,
					attn_desc,
					CUDNN_WGRAD_MODE_SET,	
					q_desc,
					devAQ,		
					k_desc,
					devAK,		
					v_desc,
					devAV,	
					o_desc,
					devDAO,
					sizeWeights,
					devWs,
					devDWs,
					sizeWkspace,
					devWkspace,
					sizeReserve,
					devReserve));
		}				
	}

	checkCudaErrors(cudaDeviceSynchronize());
	auto t2 = std::chrono::high_resolution_clock::now();

	struct kv{ 
		std::string name; 
		void * devPtr; 
		T error = 0.0; 
		int error_i = 0;
		T error_div_spread = 0.0;
		T minPyt = FLT_MAX;
		T maxPyt = FLT_MIN;
		T minCud = FLT_MAX;
		T maxCud = FLT_MIN;			
	};	
	std::vector<kv> kvvec { 
		{"q_p.weight.grad", devDWQ},
		{"k_p.weight.grad", devDWK},
		{"v_p.weight.grad", devDWV},
		{"o_p.weight.grad", devDWO},
		// {"q_p.bias.grad", devDBQ},
		// {"k_p.bias.grad", devDBK},
		// {"v_p.bias.grad", devDBV},
		// {"o_p.bias.grad", devDBO},
		// {"q_p.weight", devWQ},
		// {"k_p.weight", devWK},
		// {"v_p.weight", devWV},
		// {"o_p.weight", devWO},
		// {"q_p.bias", devBQ},
		// {"k_p.bias", devBK},
		// {"v_p.bias", devBV},
		// {"o_p.bias", devBO},		
		{"o", devAO},
	};
	if (debug_mode) {
		std::cout << "\n*******************************************************************************";
		std::cout << "\nPrinting up to the first " << FLAGS_print_limit_per_tensor << " values of each tensor:";
		std::cout << "\n*******************************************************************************\n";
	}
	std::cout << std::setprecision(4) << std::fixed;
	for (auto & [name, devPtr, error, error_i, error_div_spread, minPyt, maxPyt, minCud, maxCud] : kvvec) {
		if (cfg.vecs.find(name) == cfg.vecs.end() || cfg.vecs[name].empty() || devPtr == nullptr)
			continue;
		auto pytVec = cfg.vecs[name];
		auto size = sizeof(T) * pytVec.size();
		auto cudnnVec = pytVec;
		checkCudaErrors(cudaMemcpy(cudnnVec.data(), devPtr, size, cudaMemcpyDeviceToHost));
		if (debug_mode || FLAGS_print_accuracy_stats)
			std::cout << "\nTensor " << name << ":\n\n";		
		
		// TO DO (Maybe?): change indexing to using stride and dimension lengths 
		// info provided by cudnnGetTensorNdDescriptor and/or cudnnGetSeqDataDescriptor.

		minPyt = sizeof(T) == 8 ? DBL_MAX : FLT_MAX;
		maxPyt = sizeof(T) == 8 ? DBL_MIN : FLT_MIN;
		minCud = sizeof(T) == 8 ? DBL_MAX : FLT_MAX;
		maxCud = sizeof(T) == 8 ? DBL_MIN : FLT_MIN;	

		for (int i = 0; i < pytVec.size(); i++) {
			auto cudVal = cudnnVec[i];
			auto pytVal = pytVec[i];

			if (error < abs(cudVal - pytVal)) {
				error = abs(cudVal - pytVal);
				error_i = i;
			}
			minPyt = std::min(minPyt, pytVal);
			maxPyt = std::max(maxPyt, pytVal);														
			minCud = std::min(minCud, cudVal);
			maxCud = std::max(maxCud, cudVal);

			if (debug_mode && i < FLAGS_print_limit_per_tensor) {
				if (name == "q" || name == "k" || name == "v" || name == "o") {
					auto seq_idx = i / (batch_size * emb_dim);
					auto batch_idx = (i % (batch_size * emb_dim)) / emb_dim;
					auto emb_idx = i % emb_dim;
					std::cout << i << " "
						<< " [ seq token " << seq_idx << ", batch " << batch_idx 
						<< ", " << "emb vec component " << emb_idx << " ] ";
				} else {
					std::cout << i << " ";
				}
				std::cout << " CuDNN:"
					<< std::right << std::setw(14) << cudVal
					<< std::right << std::setw(14) << "PyTorch:" 
					<< std::right << std::setw(14) << pytVal 
					<< "\n";
			}
		}
	}	

	if (FLAGS_print_accuracy_stats) {
		std::cout << "\n*******************************************************************************";
		std::cout << "\nGreatest differences between CuDNN tensors and reference PyTorch tensors.";
		std::cout << "\n*******************************************************************************\n";
		for (auto & [name, devPtr, error, error_i, error_div_spread, minPyt, maxPyt, minCud, maxCud] : kvvec) {
			if (cfg.vecs.find(name) == cfg.vecs.end() || cfg.vecs[name].empty() || devPtr == nullptr)
				continue;		
			auto pytVec = cfg.vecs[name];
			auto size = sizeof(T) * pytVec.size();
			auto cudnnVec = pytVec;		
			checkCudaErrors(cudaMemcpy(cudnnVec.data(), devPtr, size, cudaMemcpyDeviceToHost));
			std::cout << "\nTensor " << name << ":\n\n";		
		
			std::string qkvo_idx_err_str = ""; 
			if (name == "q" || name == "k" || name == "v" || name == "o") {
				int i = error_i;
				int seq_idx_err = i / (batch_size * emb_dim);
				int batch_idx_err = (i % (batch_size * emb_dim)) / emb_dim;
				int emb_idx_err = i % emb_dim;			
				qkvo_idx_err_str = 
					" [ seq token " + std::to_string(seq_idx_err) +
					", batch " + std::to_string(batch_idx_err) +
					", emb vec comp " + std::to_string(emb_idx_err) + " ]"; 	
			}
			std::cout << std::setprecision(4) << std::fixed;
			std::cout << error << " at index " << error_i << qkvo_idx_err_str
					<< "   CuDNN: " << cudnnVec[error_i]
					<< "   PyTorch: " << pytVec[error_i] << "\n";
			std::cout << std::left << std::setw(10) << "PyTorch"
					<< "Min-Max spread: " << std::right << std::setw(10)
					<< minPyt << std::setw(5) << " to " << std::setw(10)
					<< maxPyt << "\n";
			std::cout << std::left << std::setw(10) << "CuDNN"
					<< "Min-Max spread: " << std::right << std::setw(10)
					<< minCud << std::setw(5) << " to " << std::setw(10)
					<< maxCud << "\n";					
			auto spread = std::max(maxPyt, maxCud) - std::min(minPyt, minCud);
			std::cout << std::left << std::setw(10)
				<< "Difference / spread: " << std::right << std::setw(10)
				<< (spread > 0.0 ? (100.0 * error / spread) : 0.0) << " %\n";
		}
	}

	double dur_time =
		std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() /
		1000.0f / FLAGS_iterations;
    printf("\nIteration time: %lf ms\n", dur_time);
	printf(
		"TeraFLOPS/s: %lf\n",
		2.0 * batch_size * 
		cfg.args["params_count"]
		/dur_time / 1000000000.0f);	

	cudaFree(devWs);
	cudaFree(devDWs);
	cudaFree(devWkspace);
	cudaFree(devReserve);
	cudaFree(devQSeqArray);
	cudaFree(devKSeqArray);
	cudaFree(devAQ);
	cudaFree(devAK);
	cudaFree(devAV);
	cudaFree(devAO);
	cudaFree(devDAQ);
	cudaFree(devDAK);
	cudaFree(devDAV);
	cudaFree(devDAO);
	
	std::cout<<"\n";
	return 0;
}

int main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	FLAGS_double_precision ? run<double>() : run<float>();
}
