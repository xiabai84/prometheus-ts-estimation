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

    @smart_click_strategy(retry_count=3, timeout=15)
    def robust_click(self, locator):
        """
        增强的点击方法，使用装饰器自动处理滚动和重试
        """
        print(f"执行稳健点击: {locator}")
        return True

    def login(self, url, username, password, username_field_id, password_field_id, login_button_id):
        """
        登录网站 - 使用装饰器优化点击操作
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
            
            # 使用装饰器优化的点击方法点击登录按钮
            login_button_locator = (By.ID, login_button_id)
            self.robust_click(login_button_locator)
            print("登录按钮点击完成")
            
            # 等待登录完成
            time.sleep(3)
            
        except Exception as e:
            print(f"登录过程中出现错误: {e}")
            raise

    def enhanced_login(self, url, username, password, username_field_id, password_field_id, login_button_id):
        """
        增强版登录 - 所有操作都使用稳健的方法
        """
        try:
            print(f"正在访问: {url}")
            self.driver.get(url)
            
            # 等待页面加载
            time.sleep(2)
            
            # 使用稳健的方式输入用户名
            username_locator = (By.ID, username_field_id)
            username_field = self.wait.until(
                EC.presence_of_element_located(username_locator)
            )
            username_field.clear()
            username_field.send_keys(username)
            print("用户名输入完成")
            
            # 使用稳健的方式输入密码
            password_locator = (By.ID, password_field_id)
            password_field = self.wait.until(
                EC.presence_of_element_located(password_locator)
            )
            password_field.clear()
            password_field.send_keys(password)
            print("密码输入完成")
            
            # 使用装饰器优化的点击方法点击登录按钮
            login_button_locator = (By.ID, login_button_id)
            self.robust_click(login_button_locator)
            print("登录按钮点击完成")
            
            # 等待登录完成并验证登录成功
            time.sleep(3)
            
            # 可以添加登录成功的验证逻辑
            print("登录流程完成")
            return True
            
        except Exception as e:
            print(f"登录过程中出现错误: {e}")
            return False

    def login_with_element_verification(self, url, username, password, 
                                      username_field_id, password_field_id, 
                                      login_button_id, success_selector=None):
        """
        带元素验证的登录方法
        """
        try:
            print(f"正在访问: {url}")
            self.driver.get(url)
            
            # 等待页面加载
            self.wait.until(EC.presence_of_element_located((By.ID, username_field_id)))
            
            # 输入用户名
            username_field = self.driver.find_element(By.ID, username_field_id)
            username_field.clear()
            username_field.send_keys(username)
            print("用户名输入完成")
            
            # 输入密码
            password_field = self.driver.find_element(By.ID, password_field_id)
            password_field.clear()
            password_field.send_keys(password)
            print("密码输入完成")
            
            # 使用装饰器优化的点击
            login_success = self.robust_click((By.ID, login_button_id))
            
            if login_success:
                print("登录按钮点击成功")
                
                # 等待登录完成
                time.sleep(3)
                
                # 验证登录成功
                if success_selector:
                    try:
                        success_element = self.wait.until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, success_selector))
                        )
                        print("登录成功验证通过")
                        return True
                    except:
                        print("登录成功验证失败")
                        return False
                else:
                    print("登录完成（未进行成功验证）")
                    return True
            else:
                print("登录按钮点击失败")
                return False
                
        except Exception as e:
            print(f"登录过程中出现错误: {e}")
            return False

    def close(self):
        """
        关闭浏览器
        """
        if hasattr(self, 'driver'):
            self.driver.quit()
            print("浏览器已关闭")