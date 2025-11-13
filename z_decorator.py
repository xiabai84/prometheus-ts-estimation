from functools import wraps
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def smart_click_strategy(
    scroll=True, 
    retry_count=2, 
    use_js_fallback=True,
    scroll_options=None,
    timeout=10
):
    """
    智能点击策略装饰器 - 支持类方法和 self.driver
    """
    if scroll_options is None:
        scroll_options = {'behavior': 'smooth', 'block': 'center'}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 自动提取 driver 和定位信息
            driver, element_or_locator = _extract_driver_and_locator(*args, **kwargs)
            
            for attempt in range(retry_count + 1):
                try:
                    # 获取元素对象
                    element = _get_web_element(driver, element_or_locator, timeout)
                    
                    # 如果需要滚动
                    if scroll:
                        scroll_script = f"arguments[0].scrollIntoView({scroll_options});"
                        driver.execute_script(scroll_script, element)
                        time.sleep(0.3)
                    
                    # 调用原始函数
                    result = func(*args, **kwargs)
                    
                    # 如果函数有返回值则返回，否则返回 True 表示成功
                    return result if result is not None else True
                        
                except Exception as e:
                    print(f"点击尝试 {attempt + 1} 失败: {str(e)}")
                    
                    # 最后一次尝试使用 JS 点击
                    if attempt == retry_count and use_js_fallback:
                        try:
                            element = _get_web_element(driver, element_or_locator, timeout)
                            driver.execute_script("arguments[0].click();", element)
                            return True
                        except Exception as final_error:
                            raise final_error
                    
                    time.sleep(1)
            
            return False
        return wrapper
    return decorator

def _extract_driver_and_locator(*args, **kwargs):
    """
    智能提取 driver 和定位信息
    """
    if not args:
        raise ValueError("方法缺少必要的参数")
    
    first_arg = args[0]
    driver = None
    element_or_locator = None
    
    # 情况1: 第一个参数是类实例（包含 driver 属性）
    if hasattr(first_arg, 'driver'):
        driver = first_arg.driver
        if len(args) > 1:
            element_or_locator = args[1]
        else:
            element_or_locator = kwargs.get('locator') or kwargs.get('element')
    
    # 情况2: 第一个参数是 WebDriver 实例
    elif hasattr(first_arg, 'execute_script') and callable(first_arg.execute_script):
        driver = first_arg
        if len(args) > 1:
            element_or_locator = args[1]
        else:
            element_or_locator = kwargs.get('locator') or kwargs.get('element')
    
    # 情况3: 从关键字参数获取
    if not driver and 'driver' in kwargs:
        driver = kwargs['driver']
    if not element_or_locator:
        element_or_locator = kwargs.get('locator') or kwargs.get('element')
    
    if not driver:
        raise ValueError("无法提取 driver 实例")
    
    if not element_or_locator:
        raise ValueError("无法提取元素定位信息")
    
    return driver, element_or_locator

def _get_web_element(driver, element_or_locator, timeout=10):
    """
    根据输入获取 WebElement 对象
    """
    # 如果已经是 WebElement
    if hasattr(element_or_locator, 'click') and callable(element_or_locator.click):
        return element_or_locator
    
    # 如果是定位器元组 (by, value)
    elif isinstance(element_or_locator, (tuple, list)) and len(element_or_locator) == 2:
        by, value = element_or_locator
        return WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )
    
    else:
        raise ValueError(f"不支持的定位器格式: {element_or_locator}")