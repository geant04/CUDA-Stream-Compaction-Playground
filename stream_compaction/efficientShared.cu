#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <iostream>
#include <vector>

namespace StreamCompaction {
    namespace EfficientShared {
        using StreamCompaction::Common::PerformanceTimer;

#define MAX(x, y) ((x < y) ? y : x)

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define WARP_PASSES 5
#define WARP_SCAN_SIZE MAX(BLOCK_SIZE / WARP_SIZE, 1)
#define FULL_MASK 0xFFFFFFFF

        enum ScanType : int {
            NaiveShared = 0,
            WarpShared = 1
        };

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        float getGpuTime()
        {
            return timer().getGpuElapsedTimeForPreviousOperation();
        }

        __global__ void naiveScan(int n, int stride, int *dev_odata, int *dev_idata)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n)
            {
                return;
            }

            if (index >= stride)
            {
                int out = dev_idata[index - stride] + dev_idata[index];
                dev_odata[index] = out;
            }
            else
            {
                dev_odata[index] = dev_idata[index];
            }
        }
        
        __device__ static void blockInclusiveToExclusiveInternal(int n, int *writeBuffer, int *readBuffer)
        {
            int index = threadIdx.x;
            if (index >= n)
            {
                return;
            }

            if (index == 0)
            {
                writeBuffer[index] = 0;
                return;
            }

            writeBuffer[index] = readBuffer[index - 1];
        }

        __device__ void deviceBlockInclusiveToExclusive(int n, int *writeBuffer, int *readBuffer)
        {
            blockInclusiveToExclusiveInternal(n, writeBuffer, readBuffer);
        }

        __global__ void blockInclusiveToExclusive(int n, int *writeBuffer, int *readBuffer)
        {
            blockInclusiveToExclusiveInternal(n, writeBuffer, readBuffer);
        }

        __device__ int warpInternalScan(int laneValue)
        {
            unsigned int mask = FULL_MASK;
            unsigned int logOffset;
            unsigned int laneId = threadIdx.x % WARP_SIZE;

            for ( logOffset = 0; logOffset <= 4; logOffset++ )
            {
                unsigned int delta = 1 << logOffset;
                unsigned int readValue = __shfl_up_sync(mask, laneValue, delta, WARP_SIZE);

                if (laneId >= delta)
                {
                    laneValue += readValue;
                }
            }

            return laneValue;
        }

		// writes out sum of elements in a block to blockSums array
		__global__ void blockReduce( const int n, int* dev_idata, int* blockSums)
		{
			__shared__ int warpSums[WARP_SCAN_SIZE];
			int threadId = threadIdx.x;
			int globalThreadId = threadId + blockIdx.x * BLOCK_SIZE;
			int value = ( globalThreadId < n ) ? dev_idata[globalThreadId] : 0;

            for ( int offset = WARP_SIZE / 2; offset > 0; offset >>= 1 )
            {
                value += __shfl_down_sync(FULL_MASK, value, offset);
            }

			if ( threadId % WARP_SIZE == 0 ) 
			{
				warpSums[threadId / WARP_SIZE] = value;
			}

			__syncthreads();
			
			// sum warp sums together
			if ( threadId < WARP_SIZE )
			{
				int finalSum = ( threadId < WARP_SCAN_SIZE ) ? warpSums[threadId] : 0;
				for ( int offset = WARP_SIZE / 2; offset > 0; offset >>= 1 )
				{
					finalSum += __shfl_down_sync(FULL_MASK, finalSum, offset);
				}
				if ( threadId == 0 )
				{
					blockSums[blockIdx.x] = finalSum;
				}
			}
		}

        __device__ void blockUsingWarpSumInternalScan(const int n, const int passes, 
			const bool isInclusive, int *dev_odata, int *dev_idata, int *dev_block_scanned_sums)
        {
            // Warp sum arrays
            __shared__ int warpSums[WARP_SCAN_SIZE];

			// move offset LDG up here and see if it reduces SASS stall
			const int offset = dev_block_scanned_sums ? dev_block_scanned_sums[blockIdx.x] : 0;
            int localThreadId = threadIdx.x;
            int laneId = localThreadId % WARP_SIZE;
            int warpId = localThreadId / WARP_SIZE;

            int globalThreadId = localThreadId + blockIdx.x * BLOCK_SIZE;
            int threadValue = (globalThreadId < n) ? dev_idata[globalThreadId] : 0;
            int warpScanOutput = warpInternalScan(threadValue);

            // Early return, this would occur if n <= 32
            // We would've performed the scan using intrinsics only
            if (passes <= WARP_PASSES)
            {
                if (globalThreadId < n)
                {
                    int difference = isInclusive ? 0 : threadValue;
                    dev_odata[globalThreadId] = warpScanOutput - difference;
                }
                return;
            }

            if (laneId == WARP_SIZE - 1)
            {
                warpSums[warpId] = warpScanOutput;
            }

            // Sync after warp scan results are populated
            __syncthreads();

            // Perform internal, exclusive scan on list of warp sums
            if (warpId == 0)
            {
                int warpSum = (laneId < WARP_SCAN_SIZE) ? warpSums[laneId] : 0;
                int warpSumScanOutput = warpInternalScan(warpSum);

                if (laneId < WARP_SCAN_SIZE)
                {
                    warpSums[laneId] = warpSumScanOutput;
                }
            }

            __syncthreads();

            // Add "exclusive-scan" results to the respective warps
            unsigned int difference = (warpId == 0) ? 0 : warpSums[warpId - 1];
            difference += isInclusive ? 0 : -threadValue;

            if (globalThreadId < n)
            {
                dev_odata[globalThreadId] = warpScanOutput + difference + offset;
            }
        }

        __device__ void internalScanPass(int n, int stride, int *writeBuffer, int *readBuffer)
        {
            int localThreadId = threadIdx.x;
            
            if (localThreadId < n)
            {
                int out = readBuffer[localThreadId];
                if (localThreadId >= stride)
                {
                    out += readBuffer[localThreadId - stride];
                }
                writeBuffer[localThreadId] = out;
            }
        }

		// Legacy scan code. Only here for reference of what I did before.
        __device__ void blockInternalScan(int n, int passes, int *dev_odata, int *dev_idata)
        {
            __shared__ int read[BLOCK_SIZE];
            __shared__ int write[BLOCK_SIZE];

            int localThreadId = threadIdx.x;
            read[localThreadId] = (localThreadId < n) ? dev_idata[localThreadId + blockIdx.x * blockDim.x] : 0;

            // Sync threads 1
            __syncthreads();

            int *readBuffer = read;
            int *writeBuffer = write;
            for (int pass = 0; pass < passes; pass++)
            {
                int stride = 1 << pass;
                internalScanPass(n, stride, writeBuffer, readBuffer);

                // Sync threads log(n) times
                __syncthreads();

                int *temp = readBuffer;
                readBuffer = writeBuffer;
                writeBuffer = temp;
            }

            // Add incluse/exclusive result stuff
            deviceBlockInclusiveToExclusive(n, writeBuffer, readBuffer);

            // Final write out
            dev_odata[localThreadId] = writeBuffer[localThreadId];
        }

        __global__ void efficientSharedScan(int scanType, int n, int passes,
			const bool isInclusive, int *dev_odata, int *dev_idata, int *dev_block_scanned_sums )
        {
            if (scanType == 0)
            {
                // This method only works if the number of elements can be processed by one block.
                // Exists as a stepping stone to the warp-based scan method that works for multi-block
                blockInternalScan(n, passes, dev_odata, dev_idata);
                return;
            }

            if (scanType == 1)
            {
                blockUsingWarpSumInternalScan(n, passes, isInclusive, dev_odata, dev_idata, dev_block_scanned_sums );
                return;
            }
        }

        __global__ void writeBlockSumsToArray(int n, int blocks, const bool isInclusive, int *dev_scanned_idata, int *dev_block_sum_array, int *dev_idata)
        {
            // Each thread maps to 1 block directly
            int globalThreadId = threadIdx.x + blockIdx.x * BLOCK_SIZE;
            int blockId = globalThreadId;

            if (blockId >= blocks)
            {
                return;
            }
            else
            {
                int globalBlockLastElementId = (blockId + 1) * BLOCK_SIZE - 1;
                
                // if threadId exceeds n, last element should be n-1
                globalBlockLastElementId = (globalBlockLastElementId > n ? n - 1 : globalBlockLastElementId);

                int difference = isInclusive ? 0 : dev_idata[globalBlockLastElementId];
                dev_block_sum_array[blockId] = dev_scanned_idata[globalBlockLastElementId] + difference;
            }
        }

        __global__ void addScannedSumsToBlocks(int n, int *dev_scanned_block_sum_array, int *dev_odata)
        {
            int blockId = blockIdx.x;
            int localThreadId = threadIdx.x;
            int globalThreadId = localThreadId + blockId * BLOCK_SIZE;

            if (globalThreadId >= n)
            {
                return;
            }
            else
            {
                int scannedBlockSum = dev_scanned_block_sum_array[blockId];
                dev_odata[globalThreadId] += scannedBlockSum;
            }
        }

		static int allocate_scan_upfront(int n)
		{
			int numInts = 0;
			int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
			while( blocks > 1 )
			{
				// allocate enough data for dev_block_sums and dev_block_scanned_sums
				numInts += blocks * 2;
				blocks = (blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
			}

			return numInts;
		}

        static void multi_pass_block_scan(int n, int *dev_odata, int *dev_idata, const bool isInclusive, int *dev_scratch_data)
        {
            const int passes = ilog2ceil(n);
            const int paddedArraySize = 1 << passes;
            const int blocks = (paddedArraySize + BLOCK_SIZE - 1) / BLOCK_SIZE;

            if (blocks <= 1)
            {
                // We'll just make everything inclusive by default. I guess. this is sort of rough.
                efficientSharedScan<<<blocks, BLOCK_SIZE>>>(ScanType::WarpShared, n, passes, isInclusive, dev_odata, dev_idata, nullptr );
                return;
            }

            int *dev_block_sums = dev_scratch_data;
            int *dev_block_scanned_sums = dev_scratch_data + blocks;
			int *dev_next_scratch = dev_scratch_data + 2 * blocks;

            // Gather block sums
			blockReduce<<<blocks, BLOCK_SIZE>>>(n, dev_idata, dev_block_sums);

            // Recursively perform exclusive scan on the sum, use false param
            multi_pass_block_scan(blocks, dev_block_scanned_sums, dev_block_sums, false, dev_next_scratch);

            // Propogate results back to the blocks from the dev_block_scanned_sums... results should be made exclusive but
            // to replicate slide results, we'll just make them inclusive
			// Perform scan on the block-level, in addition to gathering sums from each block
            efficientSharedScan<<<blocks, BLOCK_SIZE>>>(ScanType::WarpShared, n, passes, isInclusive, 
				dev_odata, dev_idata, dev_block_scanned_sums);
        }

        void scan_internal(const int scanType, int n, int *odata, const int *idata) {
            int* dev_idata;
            int* dev_odata;
            const int arraySize = n;
            const int passes = ilog2ceil(arraySize);
            const int paddedArraySize = 1 << passes;
            const int sizeInBytes = sizeof(int) * paddedArraySize;
			
			int *dev_scratch_data;
			const int scratchDataSizeInBytes = sizeof(int) * allocate_scan_upfront(n);

            // Memory allocation
            {
                cudaMalloc((void**)&dev_idata, sizeInBytes);
                cudaMalloc((void**)&dev_odata, sizeInBytes);
				cudaMalloc((void**)&dev_scratch_data, scratchDataSizeInBytes);
                cudaMemcpy(dev_idata, idata, sizeInBytes, cudaMemcpyHostToDevice);
                timer().startGpuTimer();
            }

            // Stream compaction algorithm/dispatches
            // 1D grid of blocks for stream compaction
            {   
                multi_pass_block_scan(arraySize, dev_odata, dev_idata, false, dev_scratch_data);
            }

            // Send results back to host + cleanup
            {
                timer().endGpuTimer();
                cudaMemcpy(odata, dev_odata, sizeInBytes, cudaMemcpyDeviceToHost);
                cudaFree(dev_idata);
                cudaFree(dev_odata);
				cudaFree(dev_scratch_data);
            }
        }

        void scan_efficient_shared_naive(int n, int *odata, const int *idata)
        {
            scan_internal(0, n, odata, idata);
        }

        void scan_efficient_warp_shared(int n, int *odata, const int *idata)
        {
            scan_internal(1, n, odata, idata);
        }
    }
}
