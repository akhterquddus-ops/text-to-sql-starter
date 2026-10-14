"""Generate the synthetic "Demo Retail Co." SQLite database.

Everything here is invented. The generator is seeded, so every run produces
the same database and the evaluation results are reproducible.

Usage:
    python -m sqlassist.data.generate            # writes data/demo.db
    python -m sqlassist.data.generate --out x.db
"""

from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SEED = 42
DEFAULT_OUT = Path("data/demo.db")

REGIONS = [
    (1, "North", "Lahore"),
    (2, "South", "Karachi"),
    (3, "Central", "Islamabad"),
    (4, "West", "Quetta"),
    (5, "East", "Multan"),
]

CATEGORIES = {
    "Electronics": [("Wireless Earbuds", 4500), ("Power Bank 20000mAh", 3800), ("Smart Watch", 9500),
                    ("Bluetooth Speaker", 6200), ("USB-C Charger", 1800)],
    "Home": [("Steel Water Bottle", 1200), ("Non-stick Pan", 3500), ("LED Desk Lamp", 2900),
             ("Bedsheet Set", 4200), ("Wall Clock", 1600)],
    "Grocery": [("Basmati Rice 5kg", 2400), ("Green Tea 100 bags", 950), ("Olive Oil 1L", 2700),
                ("Honey 500g", 1500), ("Dates 1kg", 1300)],
    "Fashion": [("Cotton Kurta", 3200), ("Leather Wallet", 2500), ("Running Shoes", 7800),
                ("Sunglasses", 2200), ("Wool Shawl", 5400)],
}

FIRST = ["Ali", "Sara", "Usman", "Ayesha", "Bilal", "Hina", "Omar", "Fatima", "Hamza", "Zainab",
         "Imran", "Mariam", "Faisal", "Noor", "Kamran", "Sana", "Tariq", "Amna", "Asad", "Rabia"]
LAST = ["Khan", "Ahmed", "Malik", "Hussain", "Sheikh", "Butt", "Qureshi", "Siddiqui", "Raza", "Chaudhry"]
SEGMENTS = ["Consumer", "Small Business", "Corporate"]
CHANNELS = ["Website", "Mobile App", "Marketplace"]
STATUSES = ["Delivered"] * 85 + ["Cancelled"] * 7 + ["Returned"] * 5 + ["Pending"] * 3

SCHEMA = """
CREATE TABLE regions (
    region_id    INTEGER PRIMARY KEY,
    region_name  TEXT NOT NULL,
    hub_city     TEXT NOT NULL
);
CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY,
    full_name    TEXT NOT NULL,
    segment      TEXT NOT NULL,
    region_id    INTEGER NOT NULL REFERENCES regions(region_id),
    signup_date  TEXT NOT NULL
);
CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category     TEXT NOT NULL,
    list_price   REAL NOT NULL,
    unit_cost    REAL NOT NULL
);
CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date   TEXT NOT NULL,
    channel      TEXT NOT NULL,
    status       TEXT NOT NULL
);
CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL,
    unit_price    REAL NOT NULL,
    discount_pct  REAL NOT NULL DEFAULT 0
);

-- Business views: the assistant is pointed at these, not the raw tables.
CREATE VIEW vw_order_lines AS
SELECT
    o.order_id,
    o.order_date,
    substr(o.order_date, 1, 7)                                   AS order_month,
    CAST(substr(o.order_date, 1, 4) AS INTEGER)                  AS order_year,
    o.channel,
    o.status,
    c.customer_id,
    c.full_name                                                  AS customer_name,
    c.segment,
    r.region_name,
    p.product_id,
    p.product_name,
    p.category,
    oi.quantity,
    oi.unit_price,
    oi.discount_pct,
    ROUND(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100.0), 2)       AS net_revenue,
    ROUND(oi.quantity * oi.unit_price * (1 - oi.discount_pct / 100.0)
          - oi.quantity * p.unit_cost, 2)                        AS gross_profit
FROM order_items oi
JOIN orders o    ON o.order_id = oi.order_id
JOIN customers c ON c.customer_id = o.customer_id
JOIN regions r   ON r.region_id = c.region_id
JOIN products p  ON p.product_id = oi.product_id;

CREATE VIEW vw_sales_by_month_region AS
SELECT
    order_month,
    order_year,
    region_name,
    COUNT(DISTINCT order_id)       AS orders,
    SUM(quantity)                  AS units,
    ROUND(SUM(net_revenue), 2)     AS net_revenue,
    ROUND(SUM(gross_profit), 2)    AS gross_profit
FROM vw_order_lines
WHERE status = 'Delivered'
GROUP BY order_month, order_year, region_name;

CREATE VIEW vw_customer_summary AS
SELECT
    c.customer_id,
    c.full_name                    AS customer_name,
    c.segment,
    r.region_name,
    c.signup_date,
    COUNT(DISTINCT CASE WHEN l.status = 'Delivered' THEN l.order_id END)      AS delivered_orders,
    ROUND(COALESCE(SUM(CASE WHEN l.status = 'Delivered' THEN l.net_revenue END), 0), 2) AS lifetime_revenue,
    MAX(CASE WHEN l.status = 'Delivered' THEN l.order_date END)              AS last_order_date
FROM customers c
JOIN regions r ON r.region_id = c.region_id
LEFT JOIN vw_order_lines l ON l.customer_id = c.customer_id
GROUP BY c.customer_id, c.full_name, c.segment, r.region_name, c.signup_date;
"""


def build(out: Path = DEFAULT_OUT) -> Path:
    rng = random.Random(SEED)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    con = sqlite3.connect(out)
    con.executescript(SCHEMA)

    con.executemany("INSERT INTO regions VALUES (?, ?, ?)", REGIONS)

    products = []
    pid = 1
    for category, items in CATEGORIES.items():
        for name, price in items:
            cost = round(price * rng.uniform(0.55, 0.75), 2)
            products.append((pid, name, category, float(price), cost))
            pid += 1
    con.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", products)

    customers = []
    start = date(2023, 1, 1)
    for cid in range(1, 401):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        segment = rng.choices(SEGMENTS, weights=[70, 20, 10])[0]
        region = rng.choices([r[0] for r in REGIONS], weights=[30, 30, 20, 8, 12])[0]
        signup = start + timedelta(days=rng.randint(0, 540))
        customers.append((cid, name, segment, region, signup.isoformat()))
    con.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers)

    orders, items = [], []
    oid, iid = 1, 1
    first_day, last_day = date(2024, 1, 1), date(2025, 12, 31)
    for _ in range(3000):
        cust = rng.choice(customers)
        signup = date.fromisoformat(cust[4])
        earliest = max(first_day, signup)
        d = earliest + timedelta(days=rng.randint(0, max(0, (last_day - earliest).days)))
        # mild seasonality: more orders in Nov/Dec
        if d.month not in (11, 12) and rng.random() < 0.12:
            d = date(d.year, rng.choice([11, 12]), rng.randint(1, 28))
        orders.append((oid, cust[0], d.isoformat(), rng.choices(CHANNELS, weights=[40, 45, 15])[0],
                       rng.choice(STATUSES)))
        for _ in range(rng.choices([1, 2, 3, 4], weights=[45, 30, 17, 8])[0]):
            p = rng.choice(products)
            qty = rng.choices([1, 2, 3, 5], weights=[60, 25, 10, 5])[0]
            disc = rng.choices([0, 5, 10, 15], weights=[60, 20, 15, 5])[0]
            items.append((iid, oid, p[0], qty, p[3], float(disc)))
            iid += 1
        oid += 1
    con.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders)
    con.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?, ?)", items)
    con.commit()
    con.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    path = build(args.out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
