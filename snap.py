from playwright.sync_api import sync_playwright
import time
import sys

def take_snap():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        
        print("Visiting Streamlit dashboard...")
        try:
            page.goto("http://localhost:8501", wait_until="networkidle")
            time.sleep(3) # Wait for initial elements
        except Exception as e:
            print(f"Failed to load dashboard: {e}")
            sys.exit(1)
            
        print("Taking standard screenshot...")
        page.screenshot(path="dashboard.png")

        print("Clicking Learning Curve tab...")
        # Find the Learning Curve tab (it's called '📉 Learning Curve')
        try:
            tab = page.locator("button", has_text="Learning Curve").first
            tab.click()
            time.sleep(2) # Give charts time to render
            page.screenshot(path="learning_curve.png")
            print("Successfully saved learning_curve.png")
        except Exception as e:
            print(f"Failed to click or capture tab: {e}")

        browser.close()

if __name__ == "__main__":
    take_snap()
