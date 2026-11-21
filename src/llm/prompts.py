"""Prompt templates. Principle: structured API fields are used directly; the LLM only fills gaps."""

PRICING_SYSTEM = (
    "You classify the pricing model of software products from their short descriptions. "
    "Reply ONLY with a JSON object."
)

PRICING_INSTRUCTION = """For each product below decide pricingModel, exactly one of:
FREE       - free to use or open source, no paid tier mentioned
FREEMIUM   - free tier plus paid upgrades / pro plans / usage limits
PAID       - paid only: subscription, one-time purchase, paid trial, per-seat pricing
ENTERPRISE - sales-led: contact sales, demo-only, custom/enterprise pricing for businesses
If the text gives no pricing signal at all, use null (do not guess).
Return {"results": [{"id": "<id>", "pricingModel": "FREE|FREEMIUM|PAID|ENTERPRISE|null"}, ...]} with one entry per product.

Products (JSON):"""

PRICING_SITE_INSTRUCTION = """For each product below you get text scraped from its own website (pricing page when available, otherwise the homepage).
Decide pricingModel from what the site actually says:
FREE       - free / open source, no paid plans
FREEMIUM   - a free plan or free tier AND paid plans / upgrades
PAID       - paid only (subscription, one-time purchase, trial that converts to paid, no free tier)
ENTERPRISE - no public prices; "contact sales", "book a demo", custom pricing for companies
Use null if the text contains no real pricing information (do not guess).
Return {"results": [{"id": "<id>", "pricingModel": "FREE|FREEMIUM|PAID|ENTERPRISE|null"}, ...]} with one entry per product.

Products (JSON):"""

PRICING_GUESS_INSTRUCTION ="""These products give no explicit pricing signal. Give your single best-guess pricingModel for each,
using the category norms (consumer apps / dev tools / AI utilities with a signup are usually FREEMIUM; open-source or
hobby projects FREE; B2B platforms sold to companies ENTERPRISE; paid apps/hardware PAID). Never return null.
Return {"results": [{"id": "<id>", "pricingModel": "FREE|FREEMIUM|PAID|ENTERPRISE"}, ...]} with one entry per product.

Products (JSON):"""
