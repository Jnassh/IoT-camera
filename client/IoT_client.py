import json
import ssl
import paho.mqtt.client as mqtt
import logging
import threading
import time
import hmac
from hashlib import sha256

def get_timestamp():
    """获取时间戳，格式为YYYYMMDDHH"""
    return time.strftime('%Y%m%d%H', time.localtime(time.time()))

def get_client_id(device_id, psw_sig_type=0):
    """生成客户端ID"""
    return f"{device_id}_0_{psw_sig_type}_{get_timestamp()}"

def get_password(secret):
    """生成密码"""
    secret_key = get_timestamp().encode('utf-8')
    secret = secret.encode('utf-8')
    return hmac.new(secret_key, secret, digestmod=sha256).hexdigest()

class IotClient(threading.Thread):
    def __init__(self, config):
        super(IotClient, self).__init__()
        self.config = config
        self.client = None
        self.logger = logging.getLogger(__name__)
        self.connected = False
        self.stop_thread = False
        self.property_set_callback = None
        self.property_get_callback = None
        
    def run(self):
        """线程运行函数"""
        self.client.loop_forever()
        
    def connect(self):
        try:
            client_id = get_client_id(self.config.device_id)
            self.client = mqtt.Client(client_id=client_id, clean_session=self.config.clean_session)
            
            # 设置用户名和密码
            try:
                password = get_password(self.config.secret)
                self.client.username_pw_set(username=self.config.device_id,
                                          password=password)
                self.logger.info("认证信息设置成功")
            except Exception as auth_error:
                self.logger.error(f"认证信息设置失败: {str(auth_error)}")
                return False
            
            # 设置回调
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect
            
            # 连接到服务器
            try:
                self.logger.info(f"正在连接到服务器 {self.config.server}:{self.config.port}")
                self.client.connect(self.config.server,
                                  self.config.port,
                                  self.config.keep_alive)
                self.logger.info("服务器连接成功")
                return True
            except Exception as conn_error:
                self.logger.error(f"服务器连接失败: {str(conn_error)}")
                return False
            
        except Exception as e:
            self.logger.error(f"连接失败: {str(e)}")
            return False
    
    def disconnect(self):
        """断开连接"""
        self.stop_thread = True
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
    
    def report_properties(self, service_properties, qos=1):
        """上报设备属性"""
        try:
            topic = f"$oc/devices/{self.config.device_id}/sys/properties/report"
            payload = {"services": service_properties}
            payload = json.dumps(payload)
            self.client.publish(topic, payload, qos)
            self.logger.debug(f"属性上报成功: {payload}")
            return True
        except Exception as e:
            self.logger.error(f"属性上报失败: {str(e)}")
            return False
    
    def set_property_set_callback(self, callback):
        """设置属性设置回调"""
        self.property_set_callback = callback
        
    def set_property_get_callback(self, callback):
        """设置属性查询回调"""
        self.property_get_callback = callback
    
    def respond_property_set(self, request_id, result_code):
        """响应平台设置属性请求"""
        try:
            topic = f"$oc/devices/{self.config.device_id}/sys/properties/set/response/request_id={request_id}"
            payload = {"result_code": 0, "result_desc": result_code}
            self.client.publish(topic, json.dumps(payload), qos=1)
        except Exception as e:
            self.logger.error(f"响应属性设置失败: {str(e)}")
    
    def respond_property_get(self, request_id, service_properties):
        """响应平台查询属性请求"""
        try:
            topic = f"$oc/devices/{self.config.device_id}/sys/properties/get/response/request_id={request_id}"
            payload = {"services": service_properties}
            self.client.publish(topic, json.dumps(payload), qos=1)
        except Exception as e:
            self.logger.error(f"响应属性查询失败: {str(e)}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            self.connected = True
            self.logger.info("已连接到IoT平台")
            # 订阅属性相关主题
            topics = [
                (f"$oc/devices/{self.config.device_id}/sys/properties/set/#", 1),
                (f"$oc/devices/{self.config.device_id}/sys/properties/get/#", 1),
                (f"$oc/devices/{self.config.device_id}/sys/commands/#", 1)
            ]
            client.subscribe(topics)
        else:
            self.connected = False
            self.logger.error(f"连接失败，返回码: {rc}")
    
    def _on_message(self, client, userdata, msg):
        """消息回调"""
        try:
            topic = msg.topic
            payload = msg.payload.decode()
            
            # 处理属性设置
            if '/sys/properties/set/' in topic:
                request_id = topic.split('request_id=')[-1]
                if self.property_set_callback:
                    self.property_set_callback(request_id, payload)
                else:
                    self.respond_property_set(request_id, 'success')
                    
            # 处理属性查询
            elif '/sys/properties/get/' in topic:
                request_id = topic.split('request_id=')[-1]
                if self.property_get_callback:
                    self.property_get_callback(request_id, payload)
                    
            self.logger.debug(f"收到消息: Topic={topic}, Payload={payload}")
        except Exception as e:
            self.logger.error(f"消息处理失败: {str(e)}")
    
    def _on_disconnect(self, client, userdata, rc):
        """断开连接回调"""
        self.connected = False
        if rc != 0:
            self.logger.warning("意外断开连接") 