# onnx_autosubgraph
onnx-subgraph tool provides  model auto partitionioning of onnx model to several sub models by operator, performance and model size limitations, 
with the order and input / output names of sub models

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
## Pre-steps
### Download the test AI models
    1. bash scripts/test_model_download.sh, then "resnet-test.onnx" will be got in ./build
    2. you can change to any other onnx files as your needs, or edit the download link in "scripts/test_model_download.sh"
### Prepare the config.json
    1. you can edit operators in "NPU_supported_ops" and "CPU_supported_ops";
    2. you can edit performance data in "performance_data" as the real HW status, 
    3. you can edit "max_subgraph_size" in case of "NPU_supported_ops" is []
    4. you can also check more examples in "config-sample-1.json" and "config-sample-2.json"
{
    "NPU_supported_ops": ["Conv", "Reshape", "Transpose", "Add", "ReduceMean", "Sub", "Div", "Mul", "Sigmoid","MatMul"],
    "CPU_supported_ops": ["Sub", "Pow", "ReduceMean", "Add", "Sqrt", "Div","Transpose", "Gather", "MatMul", "Mul", "Softmax", "Erf", "Gemm", "Conv", "Reshape",
    "Sin", "Where", "ConstantOfShape", "Cast", "Sigmoid", "Cos", "Expand", "Slice", "Unsqueeze"],
    "performance_data": [
        {"name":"Conv","CPU_time": 0.1, "NPU_time": 0.05},
        {"name":"Mul", "CPU_time": 0.15, "NPU_time": 0.07}
    ],
    "hardware_limits": {
        "max_subgraph_size": 60024.0,
        "max_subgraphs": 5
    }
 }
  
## Parse the onnx model
     ./onnx-subgraph --onnx=resnet-test.onnx
       after parsing done, subgraphs_ios.txt will be generated at current path
       
## Split the onnx model to subgraphs
    1. edit the config path and model file path at extract_onnx.py 

    2. python extract_onnx.py, after extraction done, the subgraphs will be saved at './subgraphs'
    
### Verify the subgraphs inference with original model file
    1. edit the model path, subgraph path and config path in single_vs_multiple_onnx.py

    2. edit the input shape and name of onnx model in single_vs_multiple_onnx.py

    3. compare the MSE of original inference result and subgraphs inference result
       python single_vs_multiple_onnx.py
