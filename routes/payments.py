from flask import Blueprint, request, jsonify
from config.stripe import create_checkout_session, create_subscription_session

payments = Blueprint('payments', __name__)

@payments.route('/api/create-checkout-session', methods=['POST'])
def create_checkout():
    try:
        data = request.json
        package_id = data.get('packageId')
        price_id = data.get('priceId')
        user_id = data.get('userId')

        if not all([package_id, price_id, user_id]):
            return jsonify({'error': 'Missing required fields'}), 400

        session = create_checkout_session(package_id, price_id, user_id)
        return jsonify({'url': session.url})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@payments.route('/api/create-subscription-session', methods=['POST'])
def create_subscription():
    try:
        data = request.json
        days = data.get('days')
        user_id = data.get('userId')

        if not all([days, user_id]):
            return jsonify({'error': 'Missing required fields'}), 400

        session = create_subscription_session(days, user_id)
        return jsonify({'url': session.url})

    except Exception as e:
        return jsonify({'error': str(e)}), 500 