from flask import Flask, request, jsonify

import statistics
from typing import List, Dict, Any

from routes import app

class TradingBot:
    def __init__(self):
        # Positive sentiment keywords (bullish indicators)
        self.positive_keywords = [
            'buy', 'bull', 'moon', 'pump', 'rally', 'surge', 'breakout', 'adoption',
            'institutional', 'reserve', 'etf', 'approved', 'bullish', 'rocket',
            'green', 'up', 'high', 'record', 'all-time', 'milestone', 'breakthrough',
            'positive', 'good', 'great', 'excellent', 'strong', 'massive', 'huge',
            'rising', 'gain', 'profit', 'winner', 'success', 'boom', 'soar'
        ]
        
        # Negative sentiment keywords (bearish indicators)
        self.negative_keywords = [
            'sell', 'bear', 'dump', 'crash', 'decline', 'drop', 'fall', 'correction',
            'regulation', 'ban', 'reject', 'bearish', 'red', 'down', 'low', 'bottom',
            'negative', 'bad', 'terrible', 'weak', 'collapse', 'liquidation', 'panic',
            'falling', 'loss', 'loser', 'failure', 'crater', 'plunge', 'bleeding'
        ]
        
        # High impact keywords that amplify sentiment
        self.high_impact_keywords = [
            'trump', 'bitcoin', 'btc', 'fed', 'federal', 'government', 'institutional',
            'blackrock', 'microstrategy', 'tesla', 'paypal', 'visa', 'mastercard',
            'sec', 'cftc', 'coinbase', 'binance', 'etf', 'futures', 'options',
            'whale', 'breaking', 'urgent', 'alert', 'emergency'
        ]
    
    def calculate_sentiment_score(self, title: str) -> float:
        """Calculate sentiment score based on keyword analysis"""
        title_lower = title.lower()
        
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in title_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in title_lower)
        impact_multiplier = 1 + sum(0.5 for keyword in self.high_impact_keywords if keyword in title_lower)
        
        # Base sentiment score
        sentiment = (positive_count - negative_count) * impact_multiplier
        
        # Check for caps lock (indicates urgency/importance)
        caps_ratio = sum(1 for c in title if c.isupper()) / len(title) if title else 0
        if caps_ratio > 0.3:  # More than 30% caps
            sentiment *= 1.2
        
        # Check for exclamation marks and emojis
        excitement_indicators = title.count('!') + title.count('🚀') + title.count('📈') + title.count('💰')
        sentiment += excitement_indicators * 0.1
        
        return sentiment
    
    def calculate_volume_score(self, candles: List[Dict]) -> float:
        """Calculate volume-based score"""
        if not candles or len(candles) < 2:
            return 0
        
        volumes = [candle['volume'] for candle in candles]
        
        # Calculate volume trend
        recent_volume = volumes[-1]
        avg_volume = statistics.mean(volumes[:-1]) if len(volumes) > 1 else volumes[0]
        
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # Higher volume indicates higher importance
        if volume_ratio > 2:
            return 2.0
        elif volume_ratio > 1.5:
            return 1.5
        elif volume_ratio > 1.2:
            return 1.0
        else:
            return 0.5
    
    def calculate_price_momentum(self, candles: List[Dict]) -> float:
        """Calculate price momentum from previous candles"""
        if not candles or len(candles) < 2:
            return 0
        
        # Calculate price change over the period
        start_price = candles[0]['open']
        end_price = candles[-1]['close']
        price_change = (end_price - start_price) / start_price
        
        # Calculate volatility (price range)
        price_ranges = []
        for candle in candles:
            price_range = (candle['high'] - candle['low']) / candle['open']
            price_ranges.append(price_range)
        
        avg_volatility = statistics.mean(price_ranges)
        
        # Combine momentum and volatility
        momentum_score = price_change * 10 + avg_volatility * 5
        
        return momentum_score
    
    def calculate_technical_indicators(self, candles: List[Dict]) -> Dict[str, float]:
        """Calculate basic technical indicators"""
        if not candles:
            return {'rsi': 50, 'trend': 0}
        
        closes = [candle['close'] for candle in candles]
        
        # Simple trend calculation
        if len(closes) >= 2:
            trend = (closes[-1] - closes[0]) / closes[0]
        else:
            trend = 0
        
        # Simple RSI approximation
        if len(closes) >= 3:
            gains = []
            losses = []
            for i in range(1, len(closes)):
                change = closes[i] - closes[i-1]
                if change > 0:
                    gains.append(change)
                else:
                    losses.append(abs(change))
            
            avg_gain = statistics.mean(gains) if gains else 0
            avg_loss = statistics.mean(losses) if losses else 0.01
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50
        
        return {'rsi': rsi, 'trend': trend}
    
    def calculate_event_score(self, event: Dict[str, Any]) -> float:
        """Calculate overall importance score for a news event"""
        sentiment_score = self.calculate_sentiment_score(event['title'])
        volume_score = self.calculate_volume_score(event.get('previous_candles', []))
        momentum_score = self.calculate_price_momentum(event.get('previous_candles', []))
        
        # Source reliability weight
        source_weight = 1.0
        source = event.get('source', '').lower()
        if 'twitter' in source:
            source_weight = 0.8
        elif 'reuters' in source or 'bloomberg' in source:
            source_weight = 1.5
        elif 'coindesk' in source or 'cointelegraph' in source:
            source_weight = 1.2
        
        # Combine all scores
        total_score = (abs(sentiment_score) * 3 + volume_score + abs(momentum_score)) * source_weight
        
        return total_score
    
    def make_trading_decision(self, event: Dict[str, Any]) -> str:
        """Make LONG or SHORT decision based on analysis"""
        sentiment_score = self.calculate_sentiment_score(event['title'])
        
        # Get technical indicators
        tech_indicators = self.calculate_technical_indicators(event.get('previous_candles', []))
        
        # Price momentum from previous candles
        momentum_score = self.calculate_price_momentum(event.get('previous_candles', []))
        
        # Combine sentiment with technical analysis
        decision_score = sentiment_score * 0.6 + momentum_score * 0.3 + (tech_indicators['trend'] * 10) * 0.1
        
        # RSI consideration
        rsi = tech_indicators['rsi']
        if rsi > 70:  # Overbought
            decision_score -= 0.5
        elif rsi < 30:  # Oversold
            decision_score += 0.5
        
        # Volume consideration
        volume_score = self.calculate_volume_score(event.get('previous_candles', []))
        if volume_score > 1.5:
            decision_score *= 1.1  # Amplify signal with high volume
        
        # Final decision
        if decision_score > 0:
            return 'LONG'
        else:
            return 'SHORT'
    
    def process_news_events(self, news_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process news events and return top 50 with trading decisions"""
        if not news_events:
            return []
        
        # Score all events
        scored_events = []
        for event in news_events:
            try:
                score = self.calculate_event_score(event)
                scored_events.append({
                    'event': event,
                    'score': score
                })
            except Exception as e:
                print(f"Error processing event {event.get('id', 'unknown')}: {e}")
                continue
        
        # Sort by score (highest first) and take top 50
        scored_events.sort(key=lambda x: x['score'], reverse=True)
        top_events = scored_events[:50]
        
        # Generate trading decisions
        decisions = []
        for scored_event in top_events:
            event = scored_event['event']
            try:
                decision = self.make_trading_decision(event)
                decisions.append({
                    'id': event['id'],
                    'decision': decision
                })
            except Exception as e:
                print(f"Error making decision for event {event.get('id', 'unknown')}: {e}")
                # Default to SHORT in case of error
                decisions.append({
                    'id': event['id'],
                    'decision': 'SHORT'
                })
        
        return decisions

# Initialize trading bot
trading_bot = TradingBot()

@app.route('/trading-bot', methods=['POST'])
def bot():
    try:
        # Get JSON data from request
        news_events = request.get_json()
        
        # Validate input
        if not isinstance(news_events, list):
            return jsonify({'error': 'Invalid input: expected array of news events'}), 400
        
        if len(news_events) == 0:
            return jsonify({'error': 'No news events provided'}), 400
        
        # Process events and generate decisions
        decisions = trading_bot.process_news_events(news_events)
        
        # Ensure we return exactly 50 decisions
        if len(decisions) < 50:
            # Fill remaining slots with default SHORT decisions
            existing_ids = {d['id'] for d in decisions}
            for event in news_events:
                if len(decisions) >= 50:
                    break
                if event['id'] not in existing_ids:
                    decisions.append({
                        'id': event['id'],
                        'decision': 'SHORT'
                    })
        
        # Return exactly 50 decisions
        return jsonify(decisions[:50]), 200
        
    except Exception as e:
        print(f"Error in trading endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500
