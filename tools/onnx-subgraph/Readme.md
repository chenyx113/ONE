# onnx_autosubgraph
onnx-subgraph tool provides  model auto partitionioning of onnx model to several sub models by operator, performance and model size limitations, with the order and input / output names of sub models

# How to build the onnx-subgraph
## OS environment dependence
     1. ubuntu >=20.04
     2. GCC >= 9.4.0
     3. cmake >= 3.10
     4. python >= 3.8
     5. apt-get install libprotobuf-dev protobuf-compiler

## Python packages dependence
    onnx                         1.16.0
    onnxruntime                  1.18.1
    onnxsim                      0.4.36
    torch                        2.3.1
    
## building the onnx-subgraph
    1. cd onnx-subgraph
    2. mkdir build & cd build
    3. cmake .. & make
    4. we can get following output at ./build
          ├── onnx-subgraph
          └── scripts
              ├── config.json
              ├── config-sample-1.json
              ├── config-sample-2.json
              ├── extract_onnx_lib.py
              ├── extract_onnx.py
              ├── model_inference_multiple_output.py
              ├── model_inference.py
              ├── onnx_subgraph_ut.py
              ├── quant.py
              ├── single_vs_multiple_onnx.py
              └── test_model_download.sh
# How to use the onnx-subgraph
### Pre-steps
    1. prepare the target onnx AI model, we use test.onnx for example
    2. use onnxsim to remove the complex structures before excution onnx-subgraph
    
### Parse the onnx model
    1. edit the config.json as your needs
       -> NPU_supported_ops mean operators that can be supported by NPU
       -> CPU_supported_ops mean operators that can be supported by CPU
       -> In case of operators supported by both CPU and NPU, we can describ the performance data at "performance_data"
       -> "max_subgraph_size": can set the max size of subgraph, it works only if NPU_supported_ops is NULL

    2. ./onnx-subgraph --onnx=test.onnx
       after parse done, subgraphs_ios.txt will be generated
       
 ### Split the onnx model to subgraphs
    1. edit the config path and model file path at extract_onnx.py 

    2. python extract_onnx.py, after extraction done, the subgraphs will be saved at './subgraphs'
    
### Verify the subgraphs inference with original model file
    1. edit the model path, subgraph path and config path in single_vs_multiple_onnx.py

    2. edit the input shape and name of onnx model in single_vs_multiple_onnx.py

    3. compare the MSE of original inference result and subgraphs inference result
       python single_vs_multiple_onnx.py
