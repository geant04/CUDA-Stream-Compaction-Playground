CUDA Stream Compaction Playground
======================
Originally a CIS 5650 project on testing different stream compaction and scan algorithms, I've repurposed this as my CUDA kernel practice playground. Some things I've implemented on 2^24 element tests:
- Naive "Kogge-Stone" scan (13.39ms)
- Work-efficient upsweep/downsweep scan (11.32ms)
- **Shared-memory + warp-intrinsic scan, currently at 801.98us**
    - Target is to match CUDA's thrust, which runs at ~678.94us.
    - Internally, I refer to this as "Efficient-Shared Scan".

![](img/recentResultsOct7.png)

![](img/timerResults.png)

I've added a new folder named **computeCaptures**, which contains some csv timings that I'm currently using for performance analysis/comparison.

---
Personal Info:
  * [LinkedIn](https://www.linkedin.com/in/anthonyge/), [personal website](https://www.geant.pro)
  * Tested on: Windows 11, i9-13900H @ 2600 Mhz 16GB, NVIDIA 
GeForce RTX 4070 Laptop GPU 8GB (personal)

## Parallel Algorithms Introduction
This project was initially a homework project for CIS 5650, GPU Programming at the University of Pennsylvania. I've since then reworked the base-code a bit and tested a new scan-implementation. **Profiling for warp-shared scan to come soon!**

For this assignment, I implemented several parallel algorithms in CUDA, comparing results between CPU and GPU methods. The algoriths, which I'll describe, are:
- CPU Scan
- CPU Stream Compaction
- GPU Naive Scan
- GPU Work-Efficient Scan
- GPU Stream Compaction with Scan

Under EC:
- GPU Optimized Work-Efficient Scan (EC)

We are particularly covering prefix-sum scans, where given an input array ```n``` and a binary associate operator, in our case a sum, and identity (0), we can output an array s.t each element is the result of the applying the operator on the previous element and the input element.

There are two scans we are concerned about: exclusive, which excludes element j of the input, using j-1 instead as input, and inclusive, which does.

```
Exclusive Scan:  Element j of the result does not include element j of the input:
- In:  [ 3  1  7  0  4  1  6  3]
- Out: [ 0  3  4 11 11 15 16 22]

Inclusive Scan (Prescan):  All elements including j are summed
- In:  [ 3  1  7  0  4  1  6  3]
- Out: [ 3  4 11 11 15 16 22 25]
```
*Example taken from CIS 5650 slides*

Our goal is to therefore implement different exclusive scans on GPU/CPU, and compare scalability and performance across the algorithms.

Along with scan, we also implemented stream compaction, which takes an array ```n``` and removes elements that fail a given boolean output function. In our use case, we keep elements if they are nonzero.

---
#### CPU Scan
Implementing scan on the CPU is very straightforward. We can set up a loop iterating from the 2nd element to the end, such that for element j, ```scan[j] = input[j-1] + scan[j-1]```, while also setting ```scan[0] = 0```.

In terms of analyzing base performance, we take ```O(n)``` time with ```n-1``` adds.

---
#### Naive GPU Scan
We can implement scan on the GPU using a series of passes that act on pairs of elements. For log(n) passes, we can modify our input array such that for an element j, and a given stride determined by stage number ```d``` from 0 to ```log(n)``` s.t ```strde = 2^(d-1)```, we let ```scan[j] = scan[j - stride] + scan[j]```.

The algorithm can be visualized as such in log passes:

<img src="img/naiveVisual.png" width="500"></img>
<br>
*Image taken from GPU Gems 3, 39.2.1*

Eventually, we reach our final array after all passes, and return such as the scan result.

Performance wise, this gives us ```O(nlog(n))``` adds with ```O(log(n))``` algorithmic complexity derived from the number of dispatches called.

---
#### Work-efficient GPU Scan
We can further reduce the number of adds per stage using a balanced binary tree repreesentation of our array. Using this binary tree, we can perform two "sweeps", first an upsweep (parallel reduction), and then a downsweep.

The upsweep sums up pairs of left/right nodes to create parents s.t each parent node is a sum, where we will then repeat this paired sum in continual stages until we end up with one final sum, taking log(n) stages and dispatches. Note, however, that our adds is now ```O(n)```.

<table>
  <tr>
    <th>Upsweep</th>
    <th>Downsweep</th>
  </tr>
  <tr>
    <td> <img src="img/upsweep.png" width="500"></td>
    <td> <img src="img/downsweep.png" width="500"></td>
  </tr> 
</table>
<br>

*Image taken from CIS 5650 slides*

The downsweep uses the partial sums computed from the upsweep result (observe the non-yellow cells in the above image) to reconstruct and build our final scanned array. We first clear the last element and similarly work in pairs per stage, where for each pair, we sum the right node with its left and replace the left with right.

You can notice that our total adds is also ```O(n)``` in this downsweep stage, makign the overall adds ```O(2n) = O(n)```, with ```O(n)``` swaps as well. 

While this means we have ```O(2log(n))``` dispatches, asymptotically our work-efficient performs better than our naive.

---
#### Optimized Work-efficient GPU Scan
In implementation, it is often the case that work-efficient performs worse than naive because, in turn by having less adds, we therefore have less threads doing work and therefore many inactive warps that don't need to be launched in the first place. We can dynamically optimize the blocks and block size to only launch the necessary needed number of threads. More of this is covered further in the analysis.

---
#### Efficient Shared Scan (WIP)
#### Note: I didn't get to do this for the homework originally, so this is a new feature I started in September of 2026.

Another limitation of all GPU implementations so far is the cost incurred from global memory reads and thus long scoreboard stalls.  We can avoid this issue by loading information into shared memory per SM and use a different approach that operates our scans on blocks, then merging results together.

Depending on a block size, we can parallelize by having blocks work on chunks of our input array ```n```. Per block, we can populate the array into its shared memory and then perform an inclusive sweep, while also writing overall sums into an array of block sums. We can perform an additional scan on the block sum array before (thinking of this as all sums of element before the given block), then adding the scanned block sum results back into the respective blocks, giving us an optimized scan.
 
 <table>
  <tr>
    <td> <img src="img/divideToblocks.png" width="500"></td>
    <td> <img src="img/sharedMemScan.png" width="500"></td>
  </tr> 
</table>

*Images taken from CIS 5650 slides*

While we can perform a scan on the entire block by storing all input data in shared-memory, we can instead use **warp intrinsics** to our advantage and instead perform even SMALLER scans on chunks of 32 elements, saving these warp sums to shared-memory instead and processing offset sums like above for a super optimized block scan.

```
for ( logOffset = 0; logOffset <= 4; logOffset++ )
{
    unsigned int delta = 1 << logOffset;
    unsigned int readValue = __shfl_up_sync(mask, laneValue, delta, WARP_SIZE);

    if (laneId >= delta)
    {
        laneValue += readValue;
    }
}
```

What's nice about warp-shuffling operations is that it enables us to read the registers of other threads within the same warp in lock-step - this allows us to use the shuffle_up_sync operation to effectively sum a single element owned by a thread by reading register values from other threads. Once we're done, we simply write this out to shared memory and sync once done.

More to come in this write-up soon, but we can ultimately use the algorithm above once we gather our block-scans and recursively perform a scan on top of that. A final kernel is needed to write block-increments to all values, in total resulting 2 * (N global memory reads + N global memory writes). This is because our block-scan kernel requires 2N read/writes, and our increment dispatch takes 2N read/writes.

I found that this can be reduced to 3N instead if we perform the block-sum reductions first in a separate kernel without an initial scan, similarly using wave intrinsics. 

```
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
```

We can then trivially gather block-increments from there and then perform our block-scan, merging block-increments add into that kernel. This reduces a kernel dispatch, helping our performance and improving memory throughput.

This gives us our final timings as seen in this NSight Compute capture:

![](img/efficientSharedResults.png)

This is about **~801.98us**, within 18% of thrust's results in the three kernels above. Referencing the Oct 7 LDG move CSV in computeCaptures, our scan algorithm performs **16.6x faster than Naive, and 14.1x faster than our optimized efficient.** It's pretty bananas how much we can push scan just by taking advantage of warps and shared-memory.

## Performance Analysis
**For my performance analysis, I logged the measured time 50 times per algorithm and averaged the result to get my "average runtime" stat in ms.**

### Block Size Analysis + Dynamic Block Size Optimization
#### Testing Block Sizes
To test for optimal block size, I ran my work-efficient scan on different block sizes ranging from 64 to 512.

![block comparison](img/blockComparison.png)
![block comparison](img/blockComparisonNonPow2.png)

Through some quick testing, I found that a block size of 256 found optimal results for an array size of 2^22, and similarly 2^22-3 for the non-power-of-two test. When implementing the assignment, I arbitrarily chose 128 for my block size.

A quick idea for why larger block sizes can be better is because more warps can run per block. The SM can therefore utilize more warps to hide global memory latency that often come per stage in our work-efficient scan.

---
#### Dynamic Block Size
An optimization I used in my "optimized work efficient" is to **dynamically reduce the block-size** based on the number of threads we need to run in the first place. For example, if we're at a stage in our scan that runs on 64 pairs, but my block size is 128, I dynamically match to 64. This allows less work to run on the GPU, and therefore faster kernel throughput. Using 128 can lead to immediately inactive warps, along with the added overhead of dispatching more warps too.

Here's an example snippet from my code for upsweep, notice variables ```optimizedBlocks, optimizedBlockSize```:
```
for (int d = 0; d < stages; d++)
{
    upsweep<<<optimizedBlocks, optimizedBlockSize>>>(paddedN, stride, dev_odata); // dispatch kernel using dynamic # of blocks/block size

    stride <<= 1; // double stride for next pass

    // Threads to run are halved
    threadsToRun = paddedN >> (d + 1); // in each stage, we halve threads to run. This is important!
    optimizedN = threadsToRun;

    threadsToRun = (threadsToRun >= blockSize) ? blockSize : threadsToRun; // we keep our original blockSize if our # of threads to run is greater
    threadsToRun = (threadsToRun <= 32) ? 32 : threadsToRun; // no need to dispatch < 32 threads, a warp has 32 threads

    optimizedBlockSize = threadsToRun;
    optimizedBlocks = (optimizedN + optimizedBlockSize - 1) / optimizedBlockSize;
}
```
The same idea applies for downsweep. This leads to a much better overall throughput result.

Using NSight Compute, for an example size of 2^24, we can compare the ms duration of a upsweep stage of work-efficient vs. optimized work efficient, where one threadSize is 128 vs. 64 in a case when we need to run on 64 threads:

<table>
  <tr>
    <th>Work-efficient, (131072,1,1)x(128,1,1)</th>
    <th>Optimized, dynamic block num/size</th>
  </tr>
  <tr>
    <td> <img src="img/workEfficientKernel.png" width="500"></td>
    <td> <img src="img/optimizedWorkEfficientKernel.png" width="500"></td>
  </tr> 
</table>

The speed improvements are very apparent even after 8 stages, comparing **0.17ms vs 0.03ms between our baseline work-efficient and our dynamic,** which dynamically reduces our block size based on the number of threads to compute. 

---
#### Index Remap in Upsweep/Downsweep in Optimized Work-Efficient
The above optimization is only possible by restructing our sweep kernels to avoid modulos when checking for active threads to run. Instead, **only the first # of threads to run by index are ran, ensuring that using less blocks effectively gets rid of unused warps.**

```
// Within upsweep, my logic for running a thread is as below:

int strideMult2 = stride * 2;
if (index < (n / strideMult2))
{
    // .. Do work
}

// Compare to (index % stride), re-mapping our indices that do work helps us dispatch our blocks more smartly, letting us launch less warps since we only care about index < #.
```

---

### GPU Scan Implementations Analysis
For this analysis, I compared my naive, work-efficient, optimized work-efficient, and thrust algorithms on increasingly large power-of-two array sizes. 

![scan comparison](img/scanComparison.png)
![scan comparison non pow2](img/scanComparisonNonPow2.png)
---

### CPU Perf > GPU? Huh?
Though it's hard to see in the graphs, for array sizes < (2^18), or 262,144, the CPU scan implementation consistently outperformed all GPU implementations besides thrust.

Here are the results I logged directly, measurements in ms: 
| Array Size | CPU | Naive | Work Efficient | Opt. Work Efficient |
| ---------- | -------- | ----- | -------------- | ------------- |
| 1024 (2^10)      | **0.00294**  | 0.137 | 0.163          | 0.145         |
| 4096 (2^12)      | **0.0123**   | 0.134 | 0.212          | 0.222         |
| 16384 (2^14)     | **0.0425**   | 0.201 | 0.325          | 0.248         |
| 65536 (2^16)     | **0.198**    | 0.258 | 0.368          | 0.388         |

These results can most likely be attributed to the many stalls that occur during global memory read/writes, along with the several dispatches required for the GPU implementations to work. Meanwhile, the CPU has less overhead issues and can modify information much faster.

---
### Naive > Work-efficient? Seems odd, right?
In all cases as well, the work-efficient was always slower than the naive until (2^22), when work-efficient performed better on average by 0.04ms.

| Array Size | CPU | Naive | Work Efficient | Opt. Work Efficient |
| ---------- | -------- | ----- | -------------- | ------------- |
| 4194304 (2^22)    | 11.3146  | **1.873** | **1.833**          | 0.924         |

These results obviously surprised me, as I assumed naturally that the work-efficient implementation should be faster than naive. However, this is mostly attributed to the work taken per kernel launch along with the observation that the work-efficient implementation simply launches more dispatches.

We can first observe, in NSight Compute, that every kernel ran in naive runs for roughly the same duration per dispatch. On 2^12 elements, we see each naive kernel take roughly ~2.53us duration. In total, we run for ~36.51 us for log(n) + 1 dispatchs to convert inclusive to exclusive.

![](img/naiveTimings.png)


With our work-efficient implementation which will have many unused warps due to its static block size/block number, we can interestingly note that most dispatches on the upsweep/downswep also take ~2.56us. Because of having both upsweep/downsweep, **we perform double the number of dispatches, 2 * log(n), compared to naive.** Thus, it's easier to understand why work-efficient runs worse, in total taking ~65.18us.

![](img/workEfficientTimings.png)

*There are so many dispatches that it's not even worth fitting in the image.*

It's ultimately surprising to see that our optimized work efficient perform better on average, given the circumstances and the same number of dispatches ran. It's just the case that each kernel is fast enough that the bottleneck is on kernel performance instead of # of dispatches.

---
### The General Bottleneck on GPU
Independent of calling many dispatches, the common bottlenecks in all kernels are the memory stalls from reading global memory. In NSight Compute using the Warp Stall Sampling metric (enabled in Metrics, I kind of just clicked full to get the most detail), we can see that most occurences happen during data load instructions.

![](img/naiveStall.png)

Here is a screenshot of naive showing ~41% stall sampling, easily greater than the other stalls in the kernel.

![](img/upsweepStall.png)

This screenshot is from the upsweep kernel, showing similar results.

Problems like these can therefore be circumvented using shared memory, which is much faster to access than global memory. Ideally, we would be able to quickly populated shared mem, sync threads, then only use shared mem for our kernel computes. 

<!-- ---

Using an alternative method of performing scan using shared memory by performing sub-scans on blocks, running another scan on an array of block reductions and then re-adding respective block elemens to original values in blocks, I was able to reach remarkable speedups!

![](img/optimizedSharedMem.png)

In this case, for N=2^12 (4096) **we went from 0.422ms to 0.244ms.** However I was only able to achieve this up until 2^14, before I started running into issues regarding max block size and SM memory limits. I unfortunately was not able to fix it and thus omitted this implementation in any of my performance readings, though I'm still optimistic that this can provide some nice perf wins. -->

---
#### Thrust Analysis
In all tests, thrust always outpeformed the rest of the implementations. There's not too much information to go off of for why thrust is better computationally besides the observation that it runs off of three kernels, a constant number of times, compared to the others. This probably means, then, that thrust heavily uses shared memory to run multiple stages within the kernel instead of multiple dispatches.

Using NSight Compute, I analyzed the most expensive kernel, DeviceScanKernel, with a duration of 4.22us with 1.07 compute throughput and 3.34 memory throughput. 

Digging around the kernel code, I was able to find a mention of ```AgentScan```, which does use shared memory per tile.

```
  // Shared memory for AgentScan
  __shared__ typename AgentScanT::TempStorage temp_storage;

  RealInitValueT real_init_value = init_value;

  // Process tiles
  AgentScanT(temp_storage, d_in, d_out, scan_op, real_init_value).ConsumeRange(num_items, tile_state, start_tile);
```
Thrust also probably some special indexing of memory(?), as there are mentions of loading 1D data in a **"warp-striped"** order. Looking it up online, CUDA's CUB library discusses [warp-striped](https://nvidia.github.io/cccl/cub/index.html#flexible-data-arrangement) as part of a possible speedup from processing more than one datum per thread". 

The striped arrangement, then, essentially has the items owned by a thread have a stride in-between, possibly for better coalesced access to global memory. Info about coalesced access can be found [at this link](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory) (this is actually a good time for me to read up on this too, since I was a bit lost in lecture).

![](img/stripedArrangement.png)
<br>
*Striped arrangement diagram, emphasis on thread_0. From the CUDA Core Compute Libraries*

I have no idea how thrust loads and uses warp-striped arrangement for its data, but at the end of the day, less kernels are used and there's a high chance of coalesced memory access and shared memory use that allow for better performance overall.

---
<!-- ### Test Program Output [OUDATED]

Tested output for 2^20. **Ignore the ```SHARED MEMORY scan``` since it only works for values of < 2^14.**
```
****************
** SCAN TESTS **
****************
==== cpu scan, power-of-two ====
   elapsed time: 5.4788ms    (std::chrono Measured)
    [   0   0   1   3   6  10  15  21  28  36  45  55  66 ... -2621437 -1572863 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 2.1245ms    (std::chrono Measured)
    passed
==== naive scan, power-of-two ====
   elapsed time: 1.85395ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.761792ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 1.65174ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 1.35082ms    (CUDA Measured)
    passed
==== optimized work-efficient scan, power-of-two ====
   elapsed time: 0.858496ms    (CUDA Measured)
    passed
==== optimized work-efficient SHARED MEMORY scan, power-of-two ====
   elapsed time: 1.48685ms    (CUDA Measured)
    a[256] = 32640, b[256] = 8128
    FAIL VALUE
==== optimized work-efficient scan, non-power-of-two ====
   elapsed time: 0.678656ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 1.33632ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.569504ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.8987ms    (std::chrono Measured)
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 1.1818ms    (std::chrono Measured)
    passed
==== cpu compact with scan ====
   elapsed time: 11.212ms    (std::chrono Measured)
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 1.53654ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 1.40259ms    (CUDA Measured)
    passed
```
---
For fun, here's the result for 2^12 with the shared memory working, scoring better than my optimized work-efficient by about **~0.162ms.**

```
****************
** SCAN TESTS **
****************
==== cpu scan, power-of-two ====
   elapsed time: 0.0112ms    (std::chrono Measured)
    [   0   0   1   3   6  10  15  21  28  36  45  55  66 ... 8378371 8382465 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.012ms    (std::chrono Measured)
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.57856ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.30416ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.995328ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.572416ms    (CUDA Measured)
    passed
==== optimized work-efficient scan, power-of-two ====
   elapsed time: 0.43008ms    (CUDA Measured)
    passed
==== optimized work-efficient SHARED MEMORY scan, power-of-two ====
   elapsed time: 0.268288ms    (CUDA Measured)
    passed
==== optimized work-efficient scan, non-power-of-two ====
   elapsed time: 0.428032ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.403456ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.126976ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0021ms    (std::chrono Measured)
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.002ms    (std::chrono Measured)
    passed
==== cpu compact with scan ====
   elapsed time: 0.0176ms    (std::chrono Measured)
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.35328ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.592896ms    (CUDA Measured)
    passed
``` -->
