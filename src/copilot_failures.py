import os
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try to import the new Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    logger.warning("google-genai package not installed. Running in mock mode.")

def detect_anomalous_merchants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies merchants with abnormal failure patterns.
    Filters:
    - Minimum transaction volume >= 50 (to ensure statistical significance).
    - Failure rate > 2 standard deviations above the mean failure rate of all valid merchants.
    """
    logger.info("Detecting anomalous merchants...")
    
    # Calculate per-merchant metrics
    merchant_stats = df.groupby('merchant_id').agg(
        total_tx=('transaction_id', 'count'),
        failed_tx=('status', lambda x: (x == 'failed').sum()),
        mcc=('mcc_category', lambda x: x.mode().iloc[0] if not x.dropna().empty else 'Unknown')
    ).reset_index()
    
    merchant_stats['failure_rate'] = merchant_stats['failed_tx'] / merchant_stats['total_tx']
    
    # Filter for merchants with >= 50 transactions
    stable_merchants = merchant_stats[merchant_stats['total_tx'] >= 50].copy()
    
    if len(stable_merchants) == 0:
        logger.warning("No merchants found with >= 50 transactions. Lowering limit to 10.")
        stable_merchants = merchant_stats[merchant_stats['total_tx'] >= 10].copy()
        
    mean_fail_rate = stable_merchants['failure_rate'].mean()
    std_fail_rate = stable_merchants['failure_rate'].std()
    
    cutoff = mean_fail_rate + (2 * std_fail_rate)
    
    # Flag anomalous merchants
    stable_merchants['z_score'] = (stable_merchants['failure_rate'] - mean_fail_rate) / std_fail_rate
    anomalous = stable_merchants[stable_merchants['failure_rate'] > cutoff].sort_values(by='failure_rate', ascending=False)
    
    logger.info(f"Dataset Mean Failure Rate: {mean_fail_rate:.2%}, Std Dev: {std_fail_rate:.2%}")
    logger.info(f"Anomaly Cutoff Failure Rate (Mean + 2 Std): {cutoff:.2%}")
    logger.info(f"Identified {len(anomalous)} anomalous merchants out of {len(stable_merchants)} stable merchants.")
    
    return anomalous

def get_merchant_failure_context(merchant_id: str, df: pd.DataFrame) -> dict:
    """
    Aggregates transaction and failure details for a merchant.
    This creates a compact JSON summary to pass to the LLM (minimizing token usage).
    """
    merchant_df = df[df['merchant_id'] == merchant_id].copy()
    total_tx = len(merchant_df)
    failed_df = merchant_df[merchant_df['status'] == 'failed']
    total_failed = len(failed_df)
    
    failure_rate = total_failed / total_tx if total_tx > 0 else 0.0
    mcc = merchant_df['mcc_category'].iloc[0] if total_tx > 0 else 'Unknown'
    
    # Get failure reasons distribution
    reasons = failed_df['failure_reason'].value_counts().to_dict()
    
    # Get payment methods distribution on failures
    methods = failed_df['payment_method'].value_counts().to_dict()
    
    # Get channels distribution on failures
    channels = failed_df['channel'].value_counts().to_dict()
    
    # Extract a few specific recent failure logs (max 5) for style/richness
    recent_failures = []
    for _, row in failed_df.head(5).iterrows():
        recent_failures.append({
            'timestamp': str(row['timestamp_utc']),
            'amount_egp': row['amount_egp'],
            'payment_method': row['payment_method'],
            'channel': row['channel'],
            'reason': row['failure_reason']
        })
        
    return {
        'merchant_id': merchant_id,
        'mcc_category': mcc,
        'total_transactions': total_tx,
        'failed_transactions': total_failed,
        'failure_rate_pct': round(failure_rate * 100, 2),
        'failure_reasons_breakdown': reasons,
        'failed_payment_methods': methods,
        'failed_channels': channels,
        'sample_failures': recent_failures
    }

class GeminiCopilot:
    def __init__(self, model_name: str = "gemini-3-flash"):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client_ready = False
        self.client = None
        
        if HAS_GEMINI and self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            self.client_ready = True
            logger.info("Gemini API Client configured successfully.")
        else:
            if not HAS_GEMINI:
                logger.warning("google-genai library is missing. Running in simulator mode.")
            elif not self.api_key:
                logger.warning("GEMINI_API_KEY environment variable not set. Running in simulator mode.")

    def explain_merchant_failures(self, context: dict, max_retries: int = 3) -> str:
        """
        Calls the Gemini API to explain the merchant's failure pattern.
        Includes exponential backoff retry logic for rate-limit safety (429 errors).
        Falls back through alternative models if the primary model is quota-exhausted.
        """
        system_prompt = (
            "You are a Senior Paymob Support & Integration Analyst. Your job is to analyze "
            "merchant transaction failure data, explain the dominant failure patterns in clear, "
            "non-technical business English, and recommend concrete, actionable resolutions to "
            "improve their payment success rate.\n\n"
            "Analyze the structured transaction failure statistics provided. Group failure reasons into:\n"
            "1. Client-side issues (e.g., insufficient_funds, invalid_card_details) — require merchant customer engagement.\n"
            "2. Integration/Technical issues (e.g., 3ds_authentication_failed, timeout, integration_error) — require integration checks.\n\n"
            "Write a structured support recommendation covering:\n"
            "- **Executive Summary**: 1-2 sentence overview of their transaction failure volume.\n"
            "- **Dominant Failure Patterns**: Highlighting the primary causes (percentages/counts) and what they mean.\n"
            "- **Actionable Next Steps**: Step-by-step guidance for the merchant (e.g., check frontend redirect URL, notify users of funding, verify API credentials).\n"
            "Do not hallucinate Paymob specific server logs or credentials. Keep the tone professional, helpful, and concise."
        )

        user_content = f"Merchant Failure Summary Data:\n{str(context)}"

        if not self.client_ready:
            return self._simulate_response(context)

        # Model fallback chain: try primary model, then alternatives.
        # Since 'gemini-3-flash' is not a direct API ID, map it to actual valid Gemini 3 Flash IDs.
        models_to_try = []
        if self.model_name == "gemini-3-flash":
            models_to_try.extend(["gemini-3-flash-preview"])
        else:
            models_to_try.append(self.model_name)

        # Add robust alternative fallbacks
        models_to_try.extend(["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.0-flash"])

        # Deduplicate while preserving order
        seen = set()
        models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]


        for model_name in models_to_try:
            for attempt in range(max_retries):
                try:
                    logger.info(f"Calling Gemini API model={model_name} (Attempt {attempt+1}/{max_retries}) for merchant {context['merchant_id']}...")
                    
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=user_content,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                        )
                    )
                    
                    return response.text
                    
                except Exception as e:
                    error_str = str(e)
                    logger.error(f"API call failed: {e}")
                    
                    # If quota exhausted, try next model immediately
                    if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                        if attempt < max_retries - 1:
                            # Use longer backoff for rate limits (Google suggests ~39s)
                            sleep_time = min(40, 10 * (attempt + 1))
                            logger.info(f"Rate limited. Retrying in {sleep_time} seconds...")
                            time.sleep(sleep_time)
                        else:
                            logger.warning(f"Model {model_name} quota exhausted after {max_retries} attempts. Trying next model...")
                            break  # Break inner loop to try next model
                    else:
                        if attempt < max_retries - 1:
                            sleep_time = 2 ** (attempt + 1)
                            logger.info(f"Retrying in {sleep_time} seconds...")
                            time.sleep(sleep_time)
                        else:
                            logger.warning(f"Max retries reached for model {model_name}.")
                            break

        logger.warning("All models exhausted. Falling back to simulator mode.")
        return self._simulate_response(context)

    def _simulate_response(self, context: dict) -> str:
        """
        Generates a high-quality simulated analytical report.
        Ensures the application works seamlessly without an API key.
        """
        merchant_id = context['merchant_id']
        fail_pct = context['failure_rate_pct']
        total_fail = context['failed_transactions']
        mcc = context['mcc_category']
        reasons = context['failure_reasons_breakdown']
        
        # Sort reasons to find primary issue
        sorted_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
        primary_reason = sorted_reasons[0][0] if sorted_reasons else "unknown_error"
        primary_count = sorted_reasons[0][1] if sorted_reasons else 0
        
        # Generate diagnostic text based on primary failure reason
        if primary_reason == '3ds_authentication_failed':
            pattern_analysis = (
                f"The primary driver of failures is **3DS Authentication Failures** ({primary_count} counts), "
                f"accounting for a significant share of the total {total_fail} failed attempts. This occurs when cardholders "
                "fail to complete the One-Time Passcode (OTP) screen, or when the redirect loop from the issuer page back "
                "to your success page is interrupted."
            )
            recommendations = (
                "1. **Review Frontend Redirection**: Verify that your Paymob integration correctly handles iframe-based 3DS redirects and does not block popups or issuer redirection URLs.\n"
                "2. **Optimize Payment UI**: Ensure users are clearly instructed not to close the browser or press 'Back' during 3DS loading states.\n"
                "3. **Alternative Methods**: Promote Instapay or Mobile Wallets as primary options, which have shorter checkout steps."
            )
        elif primary_reason == 'insufficient_funds':
            pattern_analysis = (
                f"The dominant failure mode is **Insufficient Funds** ({primary_count} counts). "
                "This indicates that transactions are successfully reaching the bank network, but cardholders do not "
                "have enough balance or credit limits to complete the purchase."
            )
            recommendations = (
                "1. **Introduce Installments**: Since you are in the **" + mcc + "** category, offering interest-free or consumer finance installments (e.g., Sympl, ValU) at checkout can help customers divide large transactions.\n"
                "2. **Saved Card Notifications**: Prompt users to check card balances prior to checkout, or send gentle abandoned cart emails reminding them to retry with an alternative card/wallet."
            )
        elif primary_reason == 'timeout':
            pattern_analysis = (
                f"The dominant issue is **Network Timeout** ({primary_count} counts), "
                "suggesting that requests between your server, Paymob, and the card networks are timing out. This is typical during "
                "network congestion or when API webhook listeners take too long to respond."
            )
            recommendations = (
                "1. **Review API Timeout Configuration**: Ensure your server's transaction request timeout is set to at least 30 seconds.\n"
                "2. **Asynchronous Webhooks**: Ensure your server instantly acknowledges Paymob's payment notification webhook (HTTP 200 OK) before running complex order creation queries in your database."
            )
        else:
            pattern_analysis = (
                f"The dominant failure reason is **{primary_reason}** ({primary_count} counts). "
                "This requires a technical investigation into the specific payment methods and channels experiencing issues."
            )
            recommendations = (
                "1. **Check API Keys & Integration**: Verify you are using the correct integration IDs and API keys in your production environment.\n"
                "2. **Contact Support**: Reach out to Paymob integrations team with a sample transaction ID to trace the upstream bank response code."
            )

        simulated_text = f"""### [AI] Paymob Copilot Analysis (Simulated Mode)
**Merchant ID:** {merchant_id} | **MCC Category:** {mcc}

#### 1. Executive Summary
Merchant `{merchant_id}` is experiencing an elevated transaction failure rate of **{fail_pct}%** (totaling {total_fail} failed attempts). This failure rate is statistically significant and exceeds the dataset threshold.

#### 2. Dominant Failure Patterns
* **{primary_reason.replace('_', ' ').title()}**: {pattern_analysis}
* **Payment Methods Affected**: {', '.join([f"{k} ({v})" for k, v in context['failed_payment_methods'].items()])}

#### 3. Actionable Next Steps
{recommendations}

*Note: This response was simulated because GEMINI_API_KEY was not configured or the library was missing.*
"""
        return simulated_text

if __name__ == "__main__":
    # Test script locally
    from src.data_processor import DataProcessor
    
    dp = DataProcessor()
    tx_df, _ = dp.process_all()
    
    anomalous = detect_anomalous_merchants(tx_df)
    if not anomalous.empty:
        top_anomalous_id = anomalous.iloc[0]['merchant_id']
        context = get_merchant_failure_context(top_anomalous_id, tx_df)
        print("Context generated:")
        print(context)
        
        copilot = GeminiCopilot()
        explanation = copilot.explain_merchant_failures(context)
        print("\nExplanation output:")
        print(explanation)
