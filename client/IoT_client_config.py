class IoTClientConfig:
    def __init__(self):
        # IoT平台连接参数
        self.server = "3c62575c78.st1.iotda-device.cn-east-3.myhuaweicloud.com"  # IoT平台地址
        self.port = 1883  # MQTT端口（非SSL）
        self.device_id = "6837094c94a9a05c3361f6c7_Camera_001"  # 设备ID
        self.secret = "12345678"  # 设备密钥
        
        # MQTT配置
        self.qos = 1
        self.keep_alive = 60
        self.protocol = "MQTT"  # 使用非SSL连接
        self.clean_session = True
        
        # 连接重试配置
        self.max_retries = 3
        self.retry_interval = 5  # 秒
        
        # 设备属性
        self.properties = {
            "manufacturer": "HUAWEI",
            "device_type": "Camera",
            "model": "v1.0",
            "serial_number": "123456789"
        } 