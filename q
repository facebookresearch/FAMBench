[33mcommit a1c0b28fc6bdff3564d205eedbbb1ac91ae30597[m[33m ([m[1;36mHEAD -> [m[1;32merichan1/editreadme[m[33m, [m[1;31morigin/erichan1/editreadme[m[33m)[m
Author: Eric Han <erichan1@fb.com>
Date:   Fri Oct 29 00:00:10 2021 +0000

    add bare minimum get started instructions

[33mcommit 46721c07ffe47133ec838eba4b0bce9906d98019[m
Merge: 1d152c6 254275f
Author: Sami Wilf <samiwilf@gmail.com>
Date:   Tue Oct 26 12:36:33 2021 -0600

    Merge pull request #41 from facebookresearch/samiw1/xlmr
    
    Add xlmr training loop. Include cpu-to-gpu ml data transfer in benc…

[33mcommit 254275f6fad9287ab5e5ef14c91f1c44e570344a[m[33m ([m[1;31morigin/samiw1/xlmr[m[33m)[m
Author: Sami Wilf <samiw1@fb.com>
Date:   Tue Oct 26 10:25:57 2021 -0700

    Add simple training loop. Include cpu-to-gpu ml data transfer in benchmark measurement

[33mcommit 1d152c6dc41373770476af96e11db122c4cc6915[m
Author: Sami Wilf <samiw1@fb.com>
Date:   Tue Oct 26 07:17:07 2021 -0700

    Fix README.md typos

[33mcommit 7b744e20ce7b94750f64510cadc93abf10045cf0[m
Author: Sami Wilf <samiw1@fb.com>
Date:   Tue Oct 26 07:11:58 2021 -0700

    Fix comms.py path. Make the path cwd agnostic.

[33mcommit ffdd5e6316a84fb997b079d79e6b84baf39b0ff1[m
Merge: a2aba50 06cd46f
Author: Sami Wilf <samiwilf@gmail.com>
Date:   Mon Oct 25 12:43:50 2021 -0600

    Merge pull request #34 from samiwilf/comms_driver
    
    comms_driver

[33mcommit 06cd46f4e7912834fc534b97fd069d35ade2eaa6[m
Author: samiw1 <samiw1@devgpu002.ftw6.facebook.com>
Date:   Fri Oct 22 17:07:22 2021 -0700

    Renamed driver. Changed file location. Removed par support. Added --backend cli arg

[33mcommit a2aba5099ae42eab24a62274fc40cd3d1b4b9d88[m
Merge: 38219d5 cc0b7f0
Author: Sami Wilf <samiwilf@gmail.com>
Date:   Sun Oct 24 01:23:48 2021 -0600

    Merge pull request #37 from samiwilf/torch2trt
    
    Added --use-torch2trt-for-mlp.   Removed lazy setup.   Removed  unnecessary replicate calls that reduced performance (issue currently in dlrm OSS)

[33mcommit cc0b7f09407c3ced1961ec6910f7b996c7af05a2[m
Author: Build Service <svcscm@fb.com>
Date:   Fri Oct 22 16:07:07 2021 -0700

    fixed self.parallel_model_batch_size = -1 causing prepare_parallel_model to be called once in parallel_forward. Now this no longer happens (which is what we want)

[33mcommit 38219d5760fe0eb7ff81333c2067be1524a9430c[m
Author: Aaron Shi <enye.shi@gmail.com>
Date:   Fri Oct 22 15:34:51 2021 -0700

    Initial RNNT Inference Repo (#36)
    
    This source code comes from https://github.com/mlcommons/inference/tree/master/speech_recognition/rnnt.
    With the addition of loadgen required for loading the pre-trained model.

[33mcommit 4fd2e86d61590749bdab10af5bb997a0acfbce4d[m
Author: samiw1 <samiw1@devgpu002.ftw6.facebook.com>
Date:   Thu Oct 21 21:42:32 2021 -0700

    Fixed lazy instantiation from impacting multi-gpu benchmarking. Also, fixed a bug impacting parallel inference speed. The bug is currently in dlrm OSS.

[33mcommit 306dde10025a7380cd3e9964e5335af625ef4bc5[m
Author: samiw1 <samiw1@devgpu002.ftw6.facebook.com>
Date:   Thu Oct 21 17:33:36 2021 -0700

    adding --use-torch2trt-for-mlp

[33mcommit 87cc2a9db50c7f4c1dd59af36baf4032477fe8db[m
Merge: d2dc20d 4a6488a
Author: Victor Bittorf <vsb@fb.com>
Date:   Thu Oct 21 15:41:36 2021 -0700

    Merge pull request #24 from facebookresearch/vsb1
    
    Updated readme.

[33mcommit d2dc20d7c4910d8d470b8a7fa8b42db3b24f451e[m
Author: erichan1 <30481032+erichan1@users.noreply.github.com>
Date:   Thu Oct 21 10:39:09 2021 -0700

    Create logger ability to measure batch latency (#26)
    
    * intermediate commit for batch latency
    
    * intermediate commit for batch latency. refactoring
    
    * updated result summarizer to calculate batch latency
    
    * fix to the batch info log and summarize
    
    * remove summaryrow class for now. to use later.
    
    * edit docstring and fix offby1 err
    
    * fix unsaved merge conflict resolve
    
    * minor changes
    
    * fix time_ms bug in batch_start and batch_stop

[33mcommit 2bb78e8a57cdfa1a35426631340dd50eeaa6deab[m
Author: samiw1 <samiw1@devgpu002.ftw6.facebook.com>
Date:   Thu Oct 21 09:49:36 2021 -0700

    comms_driver

[33mcommit 395ff9eb07f1c86f4d237b763a0e89bdb0b15cae[m
Author: erichan1 <30481032+erichan1@users.noreply.github.com>
Date:   Thu Oct 21 09:57:09 2021 -0700

    Fix run_stop in logger to use time from argument (#33)
    
    * fix runstop time to stop using internal time
    
    * fix again
    
    * use ns for time
    
    * remove rounding from time

[33mcommit a1be47dee2d8cf173f421bdd787848cdaecd177e[m
Author: Aaron Shi <enye.shi@gmail.com>
Date:   Wed Oct 20 17:21:28 2021 -0700

    Port Initial  RNNT OOTB Training to FB5 (#28)
    
    Add support for FB5Logger to RNN-T Training, and reduce outputs
    produced by mlperf. Add the run_rnnt_ootb_train.sh script and
    support launching for training against the LibriSpeech dataset.
    Since LibriSpeech is very large, a training session will take too long,
    so only train for 120 seconds, and save the number of samples trained.
    Also fix linting issues.

[33mcommit 4a6488afd66796c4ba4d6999afce967e1ee96bbf[m[33m ([m[1;31morigin/vsb1[m[33m)[m
Author: Victor Bittorf <vsb@fb.com>
Date:   Wed Oct 20 14:40:34 2021 -0700

    Fixing typos

[33mcommit 74dc5d29c238c013643fdcc62d7a851a02467eb3[m
Merge: 96c957e 3d20634
Author: Sami Wilf <samiwilf@gmail.com>
Date:   Wed Oct 20 15:37:32 2021 -0600

    Merge pull request #30 from samiwilf/latest-fbgemm_gpu-compatibility-required-changes
    
    latest fbgemm_gpu compatibility required changes

[33mcommit 96c957e02f1969c28edc6231f2700fb5c92cf1e5[m
Author: erichan1 <30481032+erichan1@users.noreply.github.com>
Date:   Tue Oct 19 17:33:09 2021 -0700

    port xlmr to use gpu (#31)
    
    * port xlmr to use gpu
    
    * fix to xlmr
    
    * increase nbatches
    
    * fp16 model and incr batch size
    
    * changed fb5config to famconfig and changed time to time_ns

[33mcommit 3d20634afef7a116c91c7a286001756774619c03[m
Author: samiw1 <samiw1@devgpu002.ftw6.facebook.com>
Date:   Tue Oct 19 08:40:27 2021 -0700

    changes needed to run dlrm with latest fbgemm_gpu (Oct 17 2021 commit)

[33mcommit 51e97da8d4261b0b9ae26f20b6f3652f4a0404f6[m
Author: Aaron Shi <enye.shi@gmail.com>
Date:   Fri Oct 15 14:40:20 2021 -0700

    Initial RNNT Training Repo (#27)

[33mcommit dd8de2f9b48e7d100914887dcb6102d98f0af293[m
Author: erichan1 <30481032+erichan1@users.noreply.github.com>
Date:   Thu Oct 14 13:58:00 2021 -0700

    Erichan1/xlmr bench (#16)
    
    * xlmr in progress
    
    * added xlmr initial version (not working
    
    * completed first working version of xlmr benchmark
    
    * completed v1 bare metal version of the xlmr benchmark
    
    * move unneeded functions to xlmr_extra
    
    * revert commented out lines in run_all.sh
    
    * comment nit
    
    * moved an import out of xlmr

[33mcommit 52f8de940690fec4420907f1fcf442bcada328cc[m
Merge: 6d27abb 91764c7
Author: Sami Wilf <samiwilf@gmail.com>
Date:   Thu Oct 14 11:12:00 2021 -0600

    Merge pull request #25 from samiwilf/precache-for-inference-only
    
    adding precache ability for inference-only

[33mcommit 91764c722c875689ddacfa10da9503fa4eecd425[m
Author: samiw1 <samiw1@devgpu002.ftw6.facebook.com>
Date:   Thu Oct 14 10:09:39 2021 -0700

    changed CLI arg to --precache-ml-data because the phrase is broader covering both training and test data

[33mcommit 69f49fe9dabe6cee41f54fbaa27285f5f68fa29c[m
Author: samiw1 <samiw1@devgpu002.ftw6.facebook.com>
Date:   Thu Oct 14 07:22:40 2021 -0700

    adding precache ability for inference-only

[33mcommit f68995d7184b6048e63502c7075a4279086a9eff[m
Author: Victor Bittorf <vsb@fb.com>
Date:   Wed Oct 13 16:05:01 2021 -0700

    Updated readme.

[33mcommit 6d27abb88a63d9af53bffbadaea450f371193f83[m
Author: erichan1 <30481032+erichan1@users.noreply.github.com>
Date:   Tue Oct 12 10:43:53 2021 -0700

    removed list[dict] type annotations for python < 3.9 (#23)

[33mcommit 8d1b42e14d474aa7399838cd4a7d3c08effe0e5a[m
Merge: 0ca6697 04dab38
Author: Sami Wilf <samiwilf@gmail.com>
Date:   Mon Oct 11 15:45:52 2021 -0600

    Merge pull request #13 from samiwilf/fbgemm_gpu
    
    fbgemm_gpu added; apex added; parallelized interact_features; fbgemm_gpu adds support for training & inference with 4bit,8bit, and fp16 quantized embedding tables on cpu & gpu

[33mcommit 04dab38eaf957d5ecbd3ac1e9b6f4b253cb17c9d[m
Author: samiw1 <samiw1@devgpu002.ftw6.facebook.com>
Date:   Mon Oct 11 14:29:28 2021 -0700

    fbgemm_gpu added; apex added; parallelized interact_features; fbgemm_gpu adds support for training & inference with 4bit,8bit, and fp16 quantized embedding tables on cpu & gpu

[33mcommit 0ca6697d1470edf0430327fa0d4dd775d73f787f[m
Author: erichan1 <30481032+erichan1@users.noreply.github.com>
Date:   Mon Oct 11 13:33:54 2021 -0700

    quick fix to docs (#22)
    
    * removed bench philo and added get started
    
    * shifted getting started position

[33mcommit 251cb1a2e96f3dd396855b9cc2e56fb2ebfd3784[m
Author: erichan1 <30481032+erichan1@users.noreply.github.com>
Date:   Fri Oct 8 10:13:44 2021 -0700

    Update documentation to match new run_all.sh and benchmarks  (#21)
    
    * updated documentation for getting started
    
    * added more desc to run_all.sh
    
    * further updates to using run_all.sh in docs
    
    * say run_all.sh is an example

[33mcommit d42b92961ca6bd708fd8ba2341bf3ec0a0c30847[m
Merge: d171956 236d692
Author: Sami Wilf <samiwilf@gmail.com>
Date:   Thu Oct 7 13:31:13 2021 -0600

    Merge pull request #19 from samiwilf/precached-training-data
    
    --precache-training-data flag and capability added

[33mcommit 236d692e999084bc586f1744c573f8179dfbd837[m
Merge: 6c8214f d171956
Author: Sami Wilf <samiwilf@gmail.com>
Date:   Thu Oct 7 13:30:15 2021 -0600

    Merge branch 'facebookresearch:main' into precached-training-data

[33mcommit 6c8214fb8156a05eb4d6d1981ebd5a39d116a84d[m
Author: samiw1 <samiw1@devgpu002.ftw6.facebook.com>
Date:   Thu Oct 7 12:21:26 2021 -0700

    To maintain call sequence of random numbers from randomseed, another function call is cached

[33mcommit d1719563e5ad7803e50fc048d9791ad24700ae98[m
Author: erichan1 <30481032+erichan1@users.noreply.github.com>
Date:   Thu Oct 7 12:15:37 2021 -0700

    Added support for different score metrics and units (#20)
    
    * added support for different score metrics and units
    
    * changed qps to exps

[33mcommit 44dff5272342fb4f9a6cfcca7ad06272e2985de2[m
Author: samiw1 <samiw1@devgpu002.ftw6.facebook.com>
Date:   Wed Oct 6 14:19:41 2021 -0700

    removed commented lines left in by mistake

[33mcommit 168df4b77cbe504e2fb5827cc7ba772a5fe3cc52[m
Author: samiw1 <samiw1@devgpu002.ftw6.facebook.com>
Date:   Wed Oct 6 14:09:22 2021 -0700

    --precache-training-data flag and capability added

[33mcommit 103fa371373a693a28584074f9a64c082e05865b[m
Author: erichan1 <30481032+erichan1@users.noreply.github.com>
Date:   Tue Oct 5 16:38:17 2021 -0700

    Added run_all.sh and summarizer can create [raw_view, intermediate_view] (#9)
    
    * summarizer summarizes to file with flat table structure
    
    * intermediate logging and summarizer improvement
    
    * added loggerconstants to make logging more flexible
    
    * fix another merge conflict
    
    * wrote script to run and summarize all benchmarks and polished summarizing
    
    * removed summary example
    
    * cleaned up run_all script

[33mcommit c5adeaf1b10cf12dd277e9fc1bdb6b0ac0ab910c[m
Merge: 9eb166e a8d1965
Author: nrsatish <nrsatish@users.noreply.github.com>
Date:   Tue Oct 5 16:05:54 2021 -0700

    Merge pull request #18 from nrsatish/add-ubench-configs
    
    Changes to ubenches to allow configs

[33mcommit a8d1965f4f45bc6b52ced8bc2de8de9eca1ec481[m
Author: Satish Nadathur <nrsatish@fb.com>
Date:   Tue Oct 5 14:24:41 2021 -0700

    Changes to ubenches to allow configs

[33mcommit 9eb166ed8ffcb967f7d76e2e26699fd96a35b709[m
Author: erichan1 <30481032+erichan1@users.noreply.github.com>
Date:   Tue Oct 5 11:28:07 2021 -0700

    fix logger (#17)

[33mcommit 2de650de7faa0c810d05065f1c5bbc672ccf2fc0[m
Author: erichan1 <30481032+erichan1@users.noreply.github.com>
Date:   Fri Oct 1 14:58:20 2021 -0700

    added loggerconstants to make logging more flexible (#15)

[33mcommit b35adb17ffaf249f657721a8d90ca6e8e4bb749a[m
Author: erichan1 <30481032+erichan1@users.noreply.github.com>
Date:   Mon Sep 27 14:54:13 2021 -0700

    added timing as a param to fb5logger (#14)
    
    * added timing as a param to fb5logger
    
    * add period to respond to comment

[33mcommit 20ed8bfb5860d23c38b13dd92353bb7f4368085f[m
Author: Aaron Shi <enye.shi@gmail.com>
Date:   Tue Sep 21 14:03:13 2021 -0700

    Add DLRM Inference script (#12)
    
    Similar to the run_dlrm_ootb_train.sh script,
    add run_dlrm_ootb_infer.sh script. It can be used in
    the same way as the training script. Update READMEs
    to reflect that.

[33mcommit 1cb7bcbf065c0c97e0a2775e14157a2ba4a0a6b7[m
Author: Aaron Enye Shi <enye.shi@gmail.com>
Date:   Mon Sep 20 23:57:32 2021 +0000

    DLRM.md: Clean up strange lines in doc

[33mcommit 71dd7c28769faba599c19746d8b40d9c8d045833[m
Author: erichan1 <30481032+erichan1@users.noreply.github.com>
Date:   Mon Sep 20 16:48:09 2021 -0700

    result_summarizer.py comments more descriptive of a summary (#10)

[33mcommit 5de1bb9c00e9641487e762d2e6b434fa0b3e3f9e[m
Author: Aaron Shi <enye.shi@gmail.com>
Date:   Mon Sep 20 16:46:47 2021 -0700

    Add DLRM config options and docs (#11)
    
    Adding basic DLRM config flag options, to allow users
    to swap the dlrm python flags. Update the documentation
    to reflect on these changes, and clean up.
    Co-authored DLRM.md with erichan1.

[33mcommit efc27e4bf7ad6a7d687ef88ec63079667484d2cf[m
Author: Aaron Shi <enye.shi@gmail.com>
Date:   Fri Sep 17 09:47:30 2021 -0700

    Remove unwanted files and update gitignore (#8)
    
    Added gitignore from
    https://github.com/github/gitignore/blob/master/Python.gitignore.

[33mcommit cbb2af72d9a142edfbfb38b13b869ebdd6588db4[m
Author: nrsatish <nrsatish@users.noreply.github.com>
Date:   Fri Sep 17 09:15:49 2021 -0700

    Added training compute ubenches (#4)
    
    * Added training compute ubenches
    
    * Tweaked file path names and added comms bench
    
    * Tweaked file path names and added comms bench
    
    * Added batch size

[33mcommit 8a7fdc675201bf33e5c1f2613a3eb9e255baec36[m
Author: Aaron Shi <enye.shi@gmail.com>
Date:   Thu Sep 16 15:33:10 2021 -0700

    Port initial DLRM OOTB Small to FB5 Logger (#7)
    
    For the demo, we can run a quick dlrm training
    over 200 batches, collect the log, and run the
    log summarizer to show the table.

[33mcommit f173ee7baed555bc74247db5471fd4a009572799[m
Merge: a571c8a 67d164a
Author: erichan1 <30481032+erichan1@users.noreply.github.com>
Date:   Thu Sep 16 14:02:47 2021 -0700

    Merge pull request #5 from facebookresearch/erichan1/logging
    
    Initial logging code

[33mcommit 67d164a09d25ec8488703583bfcb8d5698506718[m[33m ([m[1;31morigin/erichan1/logging[m[33m)[m
Author: Eric Han <erichan1@fb.com>
Date:   Thu Sep 16 19:08:11 2021 +0000

    added test_mllog to fb5logging

[33mcommit 8d6c728948ba08ef35c77ea70c91b25722d2281e[m
Author: Eric Han <erichan1@fb.com>
Date:   Thu Sep 16 19:06:59 2021 +0000

    renamed folder from logging to fb5logger to avoid name conflicts with stdlib

[33mcommit 2b38db0df1b1afc3533f506f623bdcecd0313bb8[m
Author: Eric Han <erichan1@fb.com>
Date:   Thu Sep 16 18:20:09 2021 +0000

    made summarizer file input optional

[33mcommit 85aca6f05bc441fed14e248c1d1cf99461bb4efe[m
Author: Eric Han <erichan1@fb.com>
Date:   Thu Sep 16 16:34:19 2021 +0000

    fixed summarizer arg parsing

[33mcommit 9910ba0ff2424d20b4a6e22ea3ecfa158247afcf[m
Author: Eric Han <erichan1@fb.com>
Date:   Thu Sep 16 16:26:07 2021 +0000

    improved logger to overwrite log files on new run and added command line args to summarizer. also added gitignore.

[33mcommit 61bfd74d967bac92f318daf4a183ac87ce7e2fa4[m
Author: Eric Han <erichan1@fb.com>
Date:   Wed Sep 15 22:41:33 2021 +0000

    initial logging codde

[33mcommit a571c8aa705c418d2310b9ca7daa5af1f6e44ba5[m
Author: Facebook Community Bot <facebook-github-bot@users.noreply.github.com>
Date:   Thu Sep 9 11:34:18 2021 -0400

    OSS Automated Fix: Addition of Code of Conduct

[33mcommit e6f08cfc93360425e2456d5ddb11e46ecbf951a7[m
Author: Facebook Community Bot <facebook-github-bot@users.noreply.github.com>
Date:   Thu Sep 9 11:31:43 2021 -0400

    OSS Automated Fix: Addition of Contributing

[33mcommit 99cdf8d5cf6bfcf36547f4815154420df6653249[m
Merge: 36cf443 1721be2
Author: nrsatish <nrsatish@users.noreply.github.com>
Date:   Fri Sep 10 10:15:21 2021 -0700

    Merge pull request #3 from nrsatish/add-param-submodule
    
    Added param submodule

[33mcommit 1721be2cccd5d6781c848b092ab5ac936192d2ad[m
Author: Satish Nadathur <nrsatish@fb.com>
Date:   Thu Sep 9 12:43:05 2021 -0700

    Added param submodule

[33mcommit 36cf4435ed9c5d025b509e88afe5d3fa98cdeebf[m[33m ([m[1;31morigin/bitfort_seed[m[33m)[m
Author: Victor Bittorf <vsb@fb.com>
Date:   Wed Sep 8 14:57:34 2021 -0700

    Seed with DLRM and readme
