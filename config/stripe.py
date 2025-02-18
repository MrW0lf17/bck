import os
import stripe
from dotenv import load_dotenv

load_dotenv()

# Initialize Stripe with your secret key
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

# Premium subscription price IDs (create these in your Stripe dashboard)
PREMIUM_PRICES = {
    1: 'price_1day',    # Replace with actual Stripe price ID for 1-day subscription
    3: 'price_3days',   # Replace with actual Stripe price ID for 3-day subscription
    7: 'price_7days',   # Replace with actual Stripe price ID for 7-day subscription
    30: 'price_30days'  # Replace with actual Stripe price ID for 30-day subscription
}

def create_checkout_session(package_id, price_id, user_id):
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': price_id,
                'quantity': 1,
            }],
            mode='payment',
            success_url=f"{os.getenv('FRONTEND_URL')}/dashboard?success=true",
            cancel_url=f"{os.getenv('FRONTEND_URL')}/pricing?canceled=true",
            metadata={
                'user_id': user_id,
                'package_id': package_id
            }
        )
        return session
    except Exception as e:
        print(f"Error creating checkout session: {str(e)}")
        raise e

def create_subscription_session(days, user_id):
    try:
        price_id = PREMIUM_PRICES.get(days)
        if not price_id:
            raise ValueError(f"Invalid subscription duration: {days} days")

        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': price_id,
                'quantity': 1,
            }],
            mode='subscription',
            success_url=f"{os.getenv('FRONTEND_URL')}/dashboard?success=true",
            cancel_url=f"{os.getenv('FRONTEND_URL')}/pricing?canceled=true",
            metadata={
                'user_id': user_id,
                'subscription_days': days
            }
        )
        return session
    except Exception as e:
        print(f"Error creating subscription session: {str(e)}")
        raise e 