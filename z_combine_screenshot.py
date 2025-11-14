from selenium import webdriver
from selenium.webdriver.common.by import By
from PIL import Image
import os

def take_and_combine_screenshots():
    driver = webdriver.Chrome()
    
    try:
        driver.get("https://www.example.com")
        driver.maximize_window()
        
        # 临时保存多个截图
        temp_files = []
        
        # 截图1：页面顶部
        driver.save_screenshot("temp_top.png")
        temp_files.append("temp_top.png")
        
        # 截图2：滚动到中间并截图
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        driver.save_screenshot("temp_middle.png")
        temp_files.append("temp_middle.png")
        
        # 截图3：滚动到底部并截图
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        driver.save_screenshot("temp_bottom.png")
        temp_files.append("temp_bottom.png")
        
        # 合并所有截图
        combine_images_vertical(temp_files, "combined_screenshot.png")
        print("多截图已合并为一个文件")
        
    finally:
        # 清理临时文件
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
        driver.quit()

def combine_images_vertical(image_paths, output_path):
    """垂直拼接多个图片"""
    images = [Image.open(path) for path in image_paths]
    
    # 计算总高度和最大宽度
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)
    
    # 创建新图片
    combined_image = Image.new('RGB', (max_width, total_height))
    
    # 拼接图片
    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height
        img.close()
    
    # 保存合并后的图片
    combined_image.save(output_path)
    combined_image.close()

if __name__ == "__main__":
    take_and_combine_screenshots()