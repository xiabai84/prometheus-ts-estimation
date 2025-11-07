from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
import time
import os

class WebAutomation:
    def __init__(self, driver_path=None):
        """
        初始化浏览器驱动
        """
        # 设置Chrome选项
        chrome_options = Options()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--start-maximized')  # 最大化窗口
        
        # 如果不需要显示浏览器界面，取消下面的注释
        # chrome_options.add_argument('--headless')
        
        try:
            if driver_path:
                service = Service(driver_path)
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                self.driver = webdriver.Chrome(options=chrome_options)
                
            self.wait = WebDriverWait(self.driver, 10)
            print("浏览器启动成功")
            
        except Exception as e:
            print(f"浏览器启动失败: {e}")
            raise

    def login(self, url, username, password, username_field_id, password_field_id, login_button_id):
        """
        登录网站
        """
        try:
            print(f"正在访问: {url}")
            self.driver.get(url)
            
            # 等待页面加载
            time.sleep(2)
            
            # 输入用户名
            username_field = self.wait.until(
                EC.presence_of_element_located((By.ID, username_field_id))
            )
            username_field.clear()
            username_field.send_keys(username)
            print("用户名输入完成")
            
            # 输入密码
            password_field = self.driver.find_element(By.ID, password_field_id)
            password_field.clear()
            password_field.send_keys(password)
            print("密码输入完成")
            
            # 点击登录按钮
            login_button = self.driver.find_element(By.ID, login_button_id)
            login_button.click()
            print("登录按钮点击完成")
            
            # 等待登录完成
            time.sleep(3)
            
            # 检查是否登录成功（可以根据实际页面调整）
            if "dashboard" in self.driver.current_url or "welcome" in self.driver.page_source.lower():
                print("登录成功！")
                return True
            else:
                print("登录可能失败，请检查")
                return False
                
        except Exception as e:
            print(f"登录过程中出错: {e}")
            return False

    def click_button(self, element_identifier, by_type=By.ID):
        """
        点击按钮或链接
        """
        try:
            button = self.wait.until(
                EC.element_to_be_clickable((by_type, element_identifier))
            )
            button.click()
            print(f"成功点击元素: {element_identifier}")
            time.sleep(2)
            return True
        except Exception as e:
            print(f"点击元素失败: {e}")
            return False

    def select_dropdown(self, dropdown_id, option_value, by_type=By.ID):
        """
        选择下拉菜单选项
        """
        try:
            dropdown = self.wait.until(
                EC.presence_of_element_located((by_type, dropdown_id))
            )
            select = Select(dropdown)
            select.select_by_value(option_value)
            print(f"下拉菜单 {dropdown_id} 选择选项: {option_value}")
            time.sleep(1)
            return True
        except Exception as e:
            print(f"选择下拉菜单失败: {e}")
            return False

    def select_dropdown_by_text(self, dropdown_id, option_text, by_type=By.ID):
        """
        通过文本选择下拉菜单选项
        """
        try:
            dropdown = self.wait.until(
                EC.presence_of_element_located((by_type, dropdown_id))
            )
            select = Select(dropdown)
            select.select_by_visible_text(option_text)
            print(f"下拉菜单 {dropdown_id} 选择文本: {option_text}")
            time.sleep(1)
            return True
        except Exception as e:
            print(f"通过文本选择下拉菜单失败: {e}")
            return False

    def wait_for_element(self, element_identifier, by_type=By.ID, timeout=10):
        """
        等待元素出现
        """
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by_type, element_identifier))
            )
            print(f"元素 {element_identifier} 已加载")
            return element
        except Exception as e:
            print(f"等待元素超时: {e}")
            return None

    def take_screenshot(self, filename="screenshot.png"):
        """
        截取屏幕截图
        """
        try:
            self.driver.save_screenshot(filename)
            print(f"截图已保存: {filename}")
        except Exception as e:
            print(f"截图失败: {e}")

    def close(self):
        """
        关闭浏览器
        """
        if self.driver:
            self.driver.quit()
            print("浏览器已关闭")

def main():
    # 配置信息 - 请根据实际网站修改这些值
    CONFIG = {
        'url': 'https://example.com/login',  # 替换为实际登录URL
        'username': 'your_username',         # 替换为实际用户名
        'password': 'your_password',         # 替换为实际密码
        'username_field_id': 'username',     # 替换为用户名输入框的ID
        'password_field_id': 'password',     # 替换为密码输入框的ID
        'login_button_id': 'login-btn',      # 替换为登录按钮的ID
        'dropdown_id': 'menu-dropdown',      # 替换为下拉菜单的ID
        'button_id': 'action-button'         # 替换为要点击的按钮ID
    }
    
    automation = None
    try:
        # 初始化自动化对象
        automation = WebAutomation()
        
        # 执行登录
        if automation.login(
            CONFIG['url'],
            CONFIG['username'],
            CONFIG['password'],
            CONFIG['username_field_id'],
            CONFIG['password_field_id'],
            CONFIG['login_button_id']
        ):
            print("开始执行页面操作...")
            
            # 示例操作1: 点击按钮
            automation.click_button(CONFIG['button_id'])
            
            # 示例操作2: 选择下拉菜单选项
            automation.select_dropdown(CONFIG['dropdown_id'], 'option_value')
            
            # 示例操作3: 通过文本选择下拉菜单
            # automation.select_dropdown_by_text(CONFIG['dropdown_id'], '选项文本')
            
            # 截取屏幕截图
            automation.take_screenshot("after_login_operations.png")
            
            print("所有操作执行完成！")
            
        else:
            print("登录失败，无法继续执行操作")
            
    except Exception as e:
        print(f"程序执行出错: {e}")
        
    finally:
        # 确保浏览器被关闭
        if automation:
            automation.close()

if __name__ == "__main__":
    main()