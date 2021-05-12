%
% MLHDN: Multiview Low-Rank Hybrid Dilated Network for SAR Target Recognition Using Limited Training Samples DEMO.
%        Version: 1.0
%        Date     : May 2021
%
%    This demo shows the MLHDN model for Multiview SAR Target Recognition.
%
%    samples.py ....... A script generating training and test meta-file before executing following experiments (RUN 1st)
%    main.py ....... A main script executing classification and analysis experiments .
%    tool.py ....... A script implementing various data manipulation functions.
%    train_test_script.py ....... A script implementing the training function, the test function, and etc.
%    model.py ....... A script implementing the HDN (MLHDN 1-view) and MLHDN (MLHDN K-view) model.
%    mvsecondpooling.py ....... A script implementing the multiview second order pooling without flatten.
%    clrbp.py ....... A script implementing Low-Rank Bilinear Pooling (LRBP) and Composite Low-Rank Bilinear Pooling (CLRBP).
% 
%    /MSTAR-10 ............... The folder including MSTAR training and test images (total 10 classes).
%    /input ............... The folder storing generated meta-file applicable to execution.
%
%   --------------------------------------
%   Note: Required core python libraries are covered
%   --------------------------------------
%   1. python 3.6.5
%   2. tensorflow-gpu 1.14.0
%   3. tensorboard 1.14.0
%   4. Keras 2.2.5
%   5. opencv-python 4.4.0.46
%   6. numpy 1.19.4
%   7. scipy 1.5.4
%   8. scikit-learn 0.23.2
%   --------------------------------------
%   Cite:
%   --------------------------------------
%
%   [1] Z. Xue and M. Zhang, "Multiview Low-Rank Hybrid Dilated Network for SAR Target Recognition Using Limited Training Samples,"
in IEEE Access, vol. 8, pp. 227847-227856, 2020, doi: 10.1109/ACCESS.2020.3046274.
%
%   --------------------------------------
%   Copyright & Disclaimer
%   --------------------------------------
%
%   The programs contained in this package are granted free of charge for
%   research and education purposes only. 
%
%   Copyright (c) 2020 by Zhaohui Xue & Mengxue Zhang
%   zhaohui.xue@hhu.edu.cn & mengxue_zhang@hhu.edu.cn
%   --------------------------------------
%   For full package:
%   --------------------------------------
%   https://sites.google.com/site/zhaohuixuers/
