import sys
import os

lib_path = os.getenv("FIBO_LIB", "")
if lib_path == "":
    print("Please set environment variable FIBO_LIB")
    sys.exit(1)

sys.path.append(lib_path)

from api_aisdk_py import api_infer_py

import functools
#from fiboaisdk.api_aisdk_py import api_infer_py
import json
import numpy as np

'''
Description:
        Python wrapper to measure a function E2E execution time

    Returns:
        Total Execution time in milliseconds
'''
# Uncomment to measure time taken for this function 
# NOTE: Additional time will be added because of "time" module
def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"{str(func)} : Elapsed time: {elapsed_time * 1000:.2f}ms")
        return value
    return wrapper_timer

'''
Description:
        Generate config file for AISDK arguments

    Returns:
        A json string
'''
def generate_config(user_values):
    config_template = {
        "name": "infer_config",
        "version": "1.0.0",
        "logger": {
            "log_level": "error",
            "log_path": "fibo_ai_sdk.log",
            "pattern": "[%Y-%m-%d %H:%M:%S.%e] [%n] [%^---%L---%$] [thread %t]:%g %# %v",
            "max_size": 1048576,
            "max_count": 10,
            "enable_console": True,
            "enable_file": False
        },
        "device": {
            "board_ssid": "board_ssid",
            "board_name": "board_name",
            "board_manufacturer": "qualcomm",
            "board_type": "board_type",
            "board_version": "board_version",
            "board_arch": "board_arch",
            "board_os": "board_os",
            "soc_name": "QCS6490",
            "soc_id": "QCS6490",
            "soc_ip": [
                {
                    "ip_name": "A72",
                    "family": "ARMv8",
                    "type": "CPU",
                    "cores": 8,
                    "frequency": 1000,
                    "l1_cache": 1024,
                    "l2_cache": 2048,
                    "l3_cache": 4096,
                    "memory": 1024
                },
                {
                    "ip_name": "Mali420",
                    "family": "Mali400",
                    "type": "GPU",
                    "cores": 4,
                    "frequency": 1000,
                    "l1_cache": 1024,
                    "l2_cache": 2048,
                    "memory": 1024
                }
            ]
        },
        "infer_engine": {
            "name": "default_session",
            "version": "1.0.0",
            "strategy": 0,
            "batch_timeout": 1000,
            "engine_num": 1,
            "priority": 0
        },
        "all_models": [
            {
                "model_name": "",
                "model_size": "",
                "version": "1.0.0",
                "model_path": "",
                "model_type": "",
                "model_cache": False,
                "model_cache_path": "",
                "run_backend": "CPU",
                "run_framework": "",
                "model_version": "1.0.0",
                "batch_size": 1
            }
        ],
        "graphs": [
            {
                "graph_name": "infer_engine",
                "version": "1.0.0",
                "graph_params": "",
                "graph_input_names": ["input"],
                "graph_input_shapes": [[-1]],
                "graph_input_types": ["float32"],
                "graph_input_layouts": ["NCHW"],
                "graph_output_names": ["output"],
                "graph_output_shapes": [[-1]],
                "graph_output_types": ["float32"],
                "graph_output_layouts": ["NCHW"],
                "all_nodes_params": {
                    "nodes": [
                        {
                            "node_name": "infer",
                            "node_type": "infer",
                            "version": "1.0.0",
                            "run_backend": "",
                            "run_framework": "",
                            "model_name": "",
                            "model_type": "all",
                            "net_type": "all",
                            "node_input_names": ["input_text"],
                            "node_input_types": ["float32"],
                            "node_input_shapes": [[-1]],
                            "node_input_layouts": ["NCHW"],
                            "node_output_names": ["tokens_output"],
                            "node_output_shapes": [[-1]],
                            "node_output_types": ["float32"],
                            "node_output_layouts": ["NCHW"],
                            "extra_node_params": ""
                        }
                    ]
                }
            }
        ],
        "application": {
            "name": "sample_api_infer",
            "version": "1.0.0",
            "description": "",
            "app_params": "",
            "input_algorithm_name": [],
            "output_algorithm_name": [],
            "all_algorithm_params": {
                "algorithms": [
                    {
                        "algorithm_name": "",
                        "version": "1.0.0",
                        "type": "",
                        "size": "",
                        "description": "",
                        "algorithm_params": "",
                        "input_graph_name": ["default_graph"],
                        "output_graph_name": ["default_graph"],
                        "all_graph_params": {}
                    }
                ]
            }
        }
    }
    
    def update_dict(d, updates):
        for key, value in updates.items():
            if isinstance(value, dict):
                d[key] = update_dict(d.get(key, {}), value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        d[key][i] = update_dict(d[key][i], item)
                    else:
                        d[key][i] = item
            else:
                d[key] = value
        return d
    
    config = update_dict(config_template, user_values)

    return json.dumps(config, indent=4)


'''
Description:
        Available Log level for FIBO AISDK
'''
class LogLevel():
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warn"
    ERROR = "error"
    CRITICAL = "critical"


'''
Description:
        Available Performance modes for SNPE runtime
'''
class PerfProfile():
    # Run in a balanced mode.
    BALANCED = 0
    # Run in high performance mode
    HIGH_PERFORMANCE = 1
    # Run in a power sensitive mode, at the expense of performance.
    POWER_SAVER = 2
    # Use system settings.  SNPE makes no calls to any performance related APIs.
    SYSTEM_SETTINGS = 3
    # Run in sustained high performance mode
    SUSTAINED_HIGH_PERFORMANCE = 4
    # Run in burst mode
    BURST = 5
    # Run in lower clock than POWER_SAVER, at the expense of performance.
    LOW_POWER_SAVER = 6
    # Run in higher clock and provides better performance than POWER_SAVER.
    HIGH_POWER_SAVER = 7
    # Run in lower balanced mode
    LOW_BALANCED = 8
    # Run in extreme power saver mode 
    EXTREME_POWERSAVER = 9


'''
Description:
        Available runtimes for model execution on Qualcomm® harwdware
'''
class Runtime():
    CPU = "CPU"
    GPU = "GPU"
    DSP = "DSP"


class SnpeContext:
    """
    Description:
        Model attributes required for running inferences.
        On instantiation, the buffer allocation, performance level and runtime is selected,
        
        Runtime Selection Order: 
            CPU (Default)
            GPU (If specifed) 
            DSP (If specifed)

        If Runtime == DSP , then need to Push Hexagon libs into the current directory.

        Args:
            dlc_path : Path to the DLC path on the device including model name like '/data/local/tmp/models/yolonas/Quant_yoloNas_s_320.dlc'.
            
            output_tensors : Output tensor/s of the DLC
            
            runtime : CPU, GPU or DSP runtime. Defaults to CPU

            profile_level : Performance level to run. Defaults to BURST
            
            log_level : Log level to output the information. Defaults to INFO.
    """
    def __init__(self, dlc_path: str = "None",
                    output_tensors : list = [],
                    runtime : str = Runtime.CPU,
                    profile_level : int = PerfProfile.BURST,
                    log_level : str = LogLevel.INFO):
        self.m_dlcpath = dlc_path 
        self.m_output_tensors = output_tensors 
        self.m_runtime = runtime
        self.profiling_level = profile_level
        self.log_level = log_level
        self.m_context = api_infer_py.InferAPI()
    
    """
    Description:
        Intializes Buffers and load network for SNPE execution

    Returns:
        SnpeContext instance
    """
    #@timer
    def Initialize(self):
        user_values = {
            "logger": {
                "log_level": self.log_level,
            },
            "all_models": [
                {
                    "model_name": self.m_dlcpath,
                    "model_path": self.m_dlcpath,
                    "run_framework": "snpe",
                    "run_backend": self.m_runtime,
                    "output_names": self.m_output_tensors,
                    "external_params": {
                        "profile_level": self.profiling_level,
                    }
                }
            ],
            "graphs": [
                {
                    "all_nodes_params": {
                        "nodes":[
                            {
                                "model_name": self.m_dlcpath,
                                "run_framework": "snpe",
                                "run_backend": self.m_runtime,
                            }
                        ]
                    }
                }
            ]
        }
        
        return self.m_context.Init(generate_config(user_values))
    
    """
    Description:
        Run inference on the target

    Returns:
        0 if execution is success; else refers to fibo-aisdk/include/fibo/error.h 
    """
    #@timer
    def Execute(self, output_names, input_feed):
        input_feed = {k: v.astype(np.float32).flatten().tolist() for k, v in input_feed.items()}
                  
        if self.m_context.Execute(input_feed) == 0:
            return self.m_context.FetchOutputs(output_names)        
        else:
            return None
    
    """
    Description:
        Release all the resource
    """
    #@timer
    def Release(self):
        return self.m_context.Release()
        