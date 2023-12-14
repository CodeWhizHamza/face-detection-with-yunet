# # import time
# # from selenium import webdriver
# # from selenium.webdriver.common.by import By

# # options = webdriver.ChromeOptions()
# # options.add_argument("--headless")

# # print("Starting browser")
# # browser = webdriver.Chrome(options=options)


# # print("Opening website")
# # browser.get(
# #     "https://www.slate.com/articles/sports/fivering_circus/2014/02/build_your_own_national_anthem_using_slate_s_interactive.html"
# # )

# # # Find the text input element by its id
# # print("Finding input fields")
# # placeInput = browser.find_element(By.ID, "input_place")
# # peoplesInput = browser.find_element(By.ID, "input_demonym")

# # print("Entering values")
# # placeInput.clear()
# # placeInput.send_keys("United States")
# # peoplesInput.clear()
# # peoplesInput.send_keys("Americans")

# # print("Submitting form")
# # browser.find_element(By.ID, "btn_generate").click()

# # print("Getting generated text")
# # generatedText = browser.find_element(By.ID, "int_generation_wrapper")

# # browser.quit()


# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.common.by import By

# driver = webdriver.Chrome()
# # options = new ChromeOptions()
# #  options.addArguments("--headless")

# driver.get(
#     "https://www.slate.com/articles/sports/fivering_circus/2014/02/build_your_own_national_anthem_using_slate_s_interactive.html"
# )

# input_element1 = driver.find_element(By.ID, "input_place")
# input_element1.clear()
# input_element1.send_keys("pakistan")
# input_element2 = driver.find_element(By.ID, "input_demonym")
# input_element2.clear()
# input_element2.send_keys("pakistan")

# generate = driver.find_element(By.ID, "btn_generate")
# generate.click()

# output_text = driver.find_element(By.ID, "int_generation_wrapper")
# output_text = output_text.text
# print(output_text)
# driver.close()
