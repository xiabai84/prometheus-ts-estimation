def retry_with_scroll(max_attempts=3, scroll_block='center'):
    """
    带重试和滚动机制的装饰器
    """
    def decorator(func):
        @wraps(func)
        def wrapper(driver, by, value, *args, **kwargs):
            locator = (by, value)
            
            for attempt in range(max_attempts):
                try:
                    # 每次重试前都确保元素可见
                    element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located(locator)
                    )
                    
                    driver.execute_script(
                        f"arguments[0].scrollIntoView({{block: '{scroll_block}'}});", 
                        element
                    )
                    time.sleep(0.5)
                    
                    return func(driver, by, value, *args, **kwargs)
                    
                except Exception as e:
                    if attempt == max_attempts - 1:  # 最后一次尝试
                        raise e
                    print(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(1)
            return None
        return wrapper
    return decorator

@retry_with_scroll(max_attempts=3, scroll_block='center')
def robust_click(driver, by, value, timeout=10):
    locator = (by, value)
    
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable(locator)
        )
        element.click()
        return True
    except:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(locator)
        )
        driver.execute_script("arguments[0].click();", element)
        return True