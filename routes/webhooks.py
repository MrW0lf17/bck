import os
import stripe
from flask import Blueprint, request, jsonify, current_app
from supabase import create_client, Client
from dotenv import load_dotenv
import logging

load_dotenv()

webhooks = Blueprint('webhooks', __name__)
logger = logging.getLogger(__name__)

def get_supabase_client():
    """Get a Supabase client instance with proper error handling."""
    supabase_url = current_app.config.get('SUPABASE_URL') or os.getenv('SUPABASE_URL')
    supabase_key = current_app.config.get('SUPABASE_KEY') or os.getenv('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        error_msg = "Missing required Supabase configuration"
        logger.error(f"{error_msg} - URL: {'Set' if supabase_url else 'Missing'}, Key: {'Set' if supabase_key else 'Missing'}")
        raise ValueError(error_msg)
    
    try:
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {str(e)}")
        raise

@webhooks.route('/api/webhooks/stripe', methods=['POST'])
def handle_stripe_webhook():
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.getenv('STRIPE_WEBHOOK_SECRET')
        )
    except ValueError as e:
        logger.error(f"Invalid payload: {str(e)}")
        return jsonify({'error': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {str(e)}")
        return jsonify({'error': 'Invalid signature'}), 400

    # Handle successful payments
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        
        try:
            supabase = get_supabase_client()
            
            # Get metadata from the session
            user_id = session['metadata'].get('user_id')
            package_id = session['metadata'].get('package_id')
            subscription_days = session['metadata'].get('subscription_days')

            if not user_id:
                logger.error("Missing user_id in session metadata")
                return jsonify({'error': 'Missing user_id'}), 400

            if package_id:  # Handle coin purchase
                # Get the coin package details
                from config.stripe import COIN_PACKAGES
                package = next((pkg for pkg in COIN_PACKAGES if pkg['id'] == package_id), None)
                
                if package:
                    total_coins = package['coins'] + package.get('bonus', 0)
                    
                    # Update user's coin balance
                    result = supabase.rpc(
                        'add_coins',
                        {
                            'p_user_id': user_id,
                            'p_amount': total_coins
                        }
                    )

                    # Record the transaction
                    supabase.table('coin_transactions').insert({
                        'user_id': user_id,
                        'amount': total_coins,
                        'transaction_type': 'purchase',
                        'metadata': {
                            'package_id': package_id,
                            'stripe_session_id': session['id']
                        }
                    }).execute()

            elif subscription_days:  # Handle premium subscription
                days = int(subscription_days)
                
                # Calculate premium end date
                from datetime import datetime, timedelta
                premium_until = datetime.now() + timedelta(days=days)

                # Update user's premium status
                supabase.auth.admin.update_user_by_id(
                    user_id,
                    {'user_metadata': {
                        'is_premium': True,
                        'premium_until': premium_until.isoformat()
                    }}
                )

                # Record the transaction
                supabase.table('subscription_transactions').insert({
                    'user_id': user_id,
                    'days': days,
                    'end_date': premium_until.isoformat(),
                    'stripe_session_id': session['id']
                }).execute()

        except Exception as e:
            logger.error(f"Error processing webhook: {str(e)}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'status': 'success'}) 