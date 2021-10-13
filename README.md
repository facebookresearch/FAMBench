# Proxy Workloads

These benchmarks represent important workloads. The faster these benchmarks are, the happier owners of important workloads are. The maintaineres, updates, and rules in this benchmark suite all exist to keep the connection between the people running these benchmarks and the people running the original workloads. 

The key things to know:
- These benchmarks are directly connected to real workloads run every day
- The main metric is throughput, subject to some constraints such as latency or max batchsize
- Data is often synthetic, though we have safeguards to ensure correctness
- There are special requirements when improving these benchmarks - it's not "anything goes"
- This incudes benchmarks (runnable on 1 device, multiple devices, clusters) and micorbenchmarks


To Get starting running the bencmark suite right away on a V100:

    TODO: Instructions here to generate the table.


## The Suite

This suite captures multiple benchmarks across, multiple devices, across multiple precisions, and includes microbenchmarks. We organize the suite so each benchmark result is identifed as:

    Benchmark = Models + Implementation + Mode + Configuration

### Models
This suite contains the following benchmarks:
- Recommendation: DLRM
- Text: XLM-R (WIP)
- Vision: CVT (Planned)
- Text: OSCAR (Planned)
- Speech: RNN-T (WIP)
- Video: Resnext-3D (Planned)
- Image: Regnet-Y (Planned)

### Implementation

Each benchmark comes in 3 different implementations:
- Out Of The Box (OOTB): Indicates the performance that is provided by the libraries and frameworks. Code is written like a regular AI engineer / research would write the code, not like a systems/hardware specialist would write the code.
- Optimized: Represents the best possible performance which can be reached; the code is tune, re-written (and perhaps even mangeld) by hardware and software experts
- Microbenchmarks: benchmarks which look at a specific component of dev, computer or cluster. These are highly unique and specialized in their purpose.

### Modes

For OOTB and optimized implementations, the modes are Inference and Training. For Microbenchmarks, the mode is the specific kind of microbenchmark being run.

### Configurations

Each implementation comes in multiple configurations. Each configfuration looks at the benchmark in a different way, such as:
- The model and data scaled to different number of devices: e.g. 1, 8, multiple node
- Different precisions and numeric formats
- Different variants of the models, representing possible different layers or sizes the model might be run at.

## Results

Running one or more benchmarks on a specific machine or cluster produces a reuslts table. Below are results for running benchmarks on a machine with a single V100 GPU.

|Model                    |Implementation|Mode        |Config             |Batch Size|Score |Units|
|-------------------------|--------------|------------|-------------------|----------|------|-----|
|Recommend: DLRM          |OOTB          |Training    |A.1dev-embed32-fp32|1024      |570.16|ex/s |
|Recommend: DLRM          |OOTB          |Inference   |A.1dev-embed4-fp32 |1024      |61.85*|ex/s |
|Recommend: DLRM          |Micro         |MLP/Linear  |linear_A.1dev      |256       |7.08  |TF/s |
|Recommend: DLRM          |Micro         |EmbeddingBag|emb_A.1dev         |65536     |537.80|GB/s |
* = missed latency target

Notice the following in this table:
- Each row is one Benchmark run a batch size (`Model + Implementation + Mode + Config` at a given batch size). More on batch size in Suite Design.
- All rows in the same table are run on the same machine. Benchmarks from different hardware must appear in different result tables.
- Some reuslts have a `*` denoting that they missed the latency target. More on latency targets in Suite Design.
- You may report multiple batch sizes for the same benchmark, they appear as different lines in the table.


### Results by System Scale
We look all the results to understand the braoder picture of performance.

** For systems that can't run the full model: ** Microbenchmarks give us a picture into potential performance and early indicators of where to explore more.

** For single device systems: ** For training, single device configurations and microbenchmarks can indicate trends in overall cluster performance; microbenchmarks run on the cluster paired with single device results can indiciate if single device performance is in fact the bottleneck. For inference, single inference is often easily parallelizable across multiple devices, the single device benchmarks are a very good indicator of real performance. This has the added advantage of being quick and easy for debugging and experiments. 

** For multiple device, single nodde: ** For Training, multidevice configurations give good insight into how single nodes perform within a cluster - this can be combined with microbenchmarks on the cluster to predict overall performance. For inference, this is a great refelction of actual workloads. This has the added advantage of being quick and easy for debugging and experiments. 

** For Clusters: ** Running these benchmarks on a cluster gives best indication of performance for Training but does not add addition information for Inference. The downside is, obviously, these runs are more costly to setup and run.


### How Results are Consumed
There are two broad comparisions that can be done: hardware-to-hardware and OOTB v. Optimized. 

- System to System: Compare two tables generated by two different systems to understand their differences
- OOTB v. Optimized: Look at one table, one system, and undersand the gap between the software (compilers, frameworks, and libraries) and what might be possible if the software was improved.

Generally, consume results is specific to the situation. Different goals will result in placing different proirities and weights when evaluating results so there isn't a one size fits all approach here. It's up to the people and situation. 


## Suite Design
We are very specific about how these benchmarks must be run and optimized in order to maintain our goal: ** improvements to these benchmarks connect directly to improvements in important internal workloads **. Where our methodology may seem arbtirary or cumbersome, it is in service of maintaining the connection to the source.

### Ownership, Versions & Updates
Each Benchmark (`Model + Implementation + Mode + Config`) is connected with an actual owner of an actual workload who endorsed the benchmark. The owner is the arbiter of changes, updates, and methodology for the benchmark. It is exceptionally frustarting to see benchmarks change while you are working on them. It sucks, and we version our benchmarks to help with book keeping. Ultimately, our goal here is to reflect the current state of what people care about - unfortunately this means (sometimes too frequently) bumping versions to ensure we are offering the best proxy to the world.

### Convergence and Accuracy
The gold standard in understand the system works is measuring convergence and accuracy of the model in the end-to-end context. Unfrotunatley, as shown by MLPerf, this is exceptionally costly, burdensome and slow. We do not place an empahsis on convergence and accuracy for the following reasons:
- We don't allow significant changes to model code (see "Improving the Benchmark Score"), so we don't expect people to be breaking convergence
- We limit the data types and percsiions to ones we understand and are known to be viable
- We (will) offer the ability to verify correctness (possibly thorugh real data or through statistical analysis on synethic data)
- We lean on benchmarks in MLPerf which has a similar suite of models and submissions to MLPerf are required to test correctness.

Overall, we aim to allow benchmarking at the granularity which is usable by people in their projects, representative of the actual workloads, and not overly cumbersome or expensive. It's a compromise.

### Data
As discussed in Convergence and Accuracy, we are not an accuracy or convergence benchmark. This frees us up to use synthetic data which significnatly improves usability and time-to-results for this suite. 

We may choose to use real data, or data dervied from real data, where we cannot generate proper synthetic data.

### Batch Sizes
Generally speaking, the bigger the batch size the better the throughput but the longer time to converge and the high the latency. Generally speaking, when running these benchmarks, people will want to see:
- The benchmark run at specific known batch sizes (where the convergce is understood) to allow for predicting and modeling
- The benchmark at the batch size which gives the best throghput, subject to either (a) a maximum batchsize for which the model will converge, or (b) a latency requirement for requests.

### Latency Limits
Inference benchmarks come with latency limits and the goal is to provide the best QPS while hitting the latency limit. Some inference benchmarks may reflect user facing operations where latency is key. Some inference benchmarks may reflect background jobs where throughput is key - so the latency limit is very high in these cases.

## Improving the Benchmark Score
The bigger the score, the better - but there are limits on how to get there. The limits depend on the implementaiton (Out-Of-The-Box OOTB, Optimized, or Microbenchmark).

- Out-Of-The-Box (OOTB): Improvements must come in through libraries, frameworks, and new hardware. No changing the model code (special exceptions for non-optimizng changes which enable porting to new hardware).
- Optimized: No holds barred - make the system shine. Just keep in mind everything you do, you're asking the actual people who run the workloads to do it too if they're going to realize that performance. You'll need to discore what changes you made, so keep track.
- Microbenchmarks - Implement the same operation as defined, and make it as fast as possible.
