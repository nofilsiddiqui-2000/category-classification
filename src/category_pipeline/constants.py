"""Static definitions for super-category classification."""

from __future__ import annotations

SUPER_CATEGORIES = ["Food", "Drinks", "Home Care", "Personal Care", "Other"]

CATEGORY_DEFINITIONS = {
    "Food": (
        "All edible goods for household nourishment: cereals, meat, fish, milk,"
        " dairy, eggs, oils, fats, fruits, nuts, vegetables, sugar, confectionery,"
        " desserts, sauces, condiments, spices, and culinary herbs."
    ),
    "Drinks": (
        "All beverages purchased by the household, whether consumed at home or"
        " elsewhere, including alcoholic and non-alcoholic drinks."
    ),
    "Home Care": (
        "Household cleaning/maintenance products and related non-personal-use"
        " consumables such as detergents, bleaches, disinfectants, laundry products,"
        " floor/window cleaners, drain products, household gloves, cleaning tools,"
        " garbage/storage bags, disposable household items, and similar household-use"
        " utility products."
    ),
    "Personal Care": (
        "Personal hygiene, grooming, beauty, and body-care products and tools:"
        " soaps, oral care, shaving products, deodorants, skincare, haircare,"
        " cosmetics, and personal hygiene articles."
    ),
    "Other": (
        "Use only when the category does not belong to Food, Drinks, Home Care,"
        " or Personal Care."
    ),
}

DEFAULT_CATEGORY_COLUMN_CANDIDATES = [
    "Category_Names",
    "Category Name",
    "Category",
    "category_name",
    "category",
    "name",
]

LABEL_NORMALIZATION = {
    "food": "Food",
    "drinks": "Drinks",
    "drink": "Drinks",
    "home care": "Home Care",
    "homecare": "Home Care",
    "personal care": "Personal Care",
    "personalcare": "Personal Care",
    "other": "Other",
}
