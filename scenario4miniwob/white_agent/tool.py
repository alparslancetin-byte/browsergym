# tool.py
from playwright.sync_api import Page
from typing import Tuple, Optional

# Return format for every helper:
# (action_string, log_message)

# Action Execute tools

def click_text(page: Page, text: str) -> Tuple[str, str]:
    page.get_by_text(text).click()
    return (
        f"page.get_by_text('{text}').click()",
        f"Clicked text: {text}"
    )


def click_role(page: Page, role: str, name: str) -> Tuple[str, str]:
    page.get_by_role(role, name=name).click()
    return (
        f"page.get_by_role('{role}', name='{name}').click()",
        f"Clicked role '{role}' with name '{name}'"
    )


def type_into(page: Page, selector: str, content: str) -> Tuple[str, str]:
    box = page.locator(selector)
    box.click()
    box.fill(content)
    return (
        f"page.locator('{selector}').fill('{content}')",
        f"Typed into {selector}: {content}"
    )


def wait_and_click(page: Page, selector: str) -> Tuple[str, str]:
    page.wait_for_selector(selector)
    page.locator(selector).click()
    return (
        f"page.wait_for_selector('{selector}'); page.locator('{selector}').click()",
        f"Waited and clicked {selector}"
    )


def select_dropdown(page: Page, selector: str, option: str) -> Tuple[str, str]:
    page.locator(selector).select_option(option)
    return (
        f"page.locator('{selector}').select_option('{option}')",
        f"Selected option {option} from {selector}"
    )


def scroll_to_bottom(page: Page) -> Tuple[str, str]:
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    return (
        "page.evaluate('window.scrollTo(0, document.body.scrollHeight)')",
        "Scrolled to bottom"
    )


def exists(page: Page, selector: str) -> bool:
    return page.locator(selector).count() > 0


def safe_click(page: Page, selector: str) -> Tuple[Optional[str], str]:
    if exists(page, selector):
        page.locator(selector).click()
        return (
            f"page.locator('{selector}').click()",
            f"Safely clicked {selector}"
        )
    return (None, f"{selector} not found, no click executed")


def smart_click(page: Page, keyword: str) -> Tuple[Optional[str], str]:
    candidates = [
        page.get_by_text(keyword),
        page.get_by_role("button", name=keyword),
        page.get_by_placeholder(keyword),
        page.get_by_label(keyword)
    ]

    for c in candidates:
        if c.count() > 0:
            c.click()
            return (
                f"[smart_click] Clicked element matching '{keyword}'",
                f"Smart click matched and clicked: {keyword}"
            )

    return (None, f"No element matched keyword: {keyword}")


def retry(page: Page, selector: str, tries: int = 3) -> Tuple[bool, str]:
    for i in range(tries):
        try:
            page.locator(selector).click()
            return (
                True,
                f"Retry click success on {selector} after {i+1} attempt(s)"
            )
        except:
            page.wait_for_timeout(150)

    return (False, f"Retry click failed for {selector}")
