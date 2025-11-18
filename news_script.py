import csv
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import google.generativeai as genai
import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class NewsArticle:
    title: str
    description: str
    source: str
    url: str
    published_at: str
    content: str = ""
    ticker: Optional[str] = None


@dataclass
class SentimentAnalysis:
    """Data class for sentiment analysis results"""

    ticker: str
    news_date: str
    headline: str
    sentiment_score: float  # -1 to 1
    summary: str
    confidence: float
    key_topics: List[str]
    market_impact: str
    raw_response: Dict[str, Any]


class StockTickerExtractor:

    # Common stock tickers for detection
    COMMON_TICKERS = {
        "AAPL": "Apple",
        "GOOGL": "Google",
        "MSFT": "Microsoft",
        "AMZN": "Amazon",
        "TSLA": "Tesla",
        "META": "Meta",
        "NVDA": "Nvidia",
        "JPM": "JPMorgan",
        "BAC": "Bank of America",
        "WMT": "Walmart",
        "JNJ": "Johnson & Johnson",
        "V": "Visa",
        "PG": "Procter & Gamble",
        "UNH": "UnitedHealth",
        "MA": "Mastercard",
        "HD": "Home Depot",
        "DIS": "Disney",
        "NFLX": "Netflix",
        "PYPL": "PayPal",
        "ADBE": "Adobe",
        "CRM": "Salesforce",
        "PFE": "Pfizer",
        "NKE": "Nike",
        "INTC": "Intel",
        "AMD": "AMD",
        "BA": "Boeing",
        "GS": "Goldman Sachs",
        "MS": "Morgan Stanley",
        "C": "Citigroup",
        "WFC": "Wells Fargo",
        "IBM": "IBM",
        "GE": "General Electric",
        "F": "Ford",
        "GM": "General Motors",
        "XOM": "Exxon",
        "CVX": "Chevron",
        "COST": "Costco",
        "SBUX": "Starbucks",
        "TGT": "Target",
        "UBER": "Uber",
        "SQ": "Square",
        "SPOT": "Spotify",
        "SNAP": "Snap",
        "TWTR": "Twitter",
        "ZM": "Zoom",
        "ROKU": "Roku",
        "PLTR": "Palantir",
        "COIN": "Coinbase",
    }

    @classmethod
    def extract_ticker(
        cls, title: str, content: str, gemini_response: Dict = None
    ) -> str:

        # First check if Gemini identified a ticker
        if gemini_response and "ticker" in gemini_response:
            ticker = gemini_response.get("ticker", "").upper()
            if ticker and ticker != "GENERAL":
                return ticker

        text = f"{title} {content}".upper()

        ticker_pattern = (
            r"\b([A-Z]{1,5})\b(?:\s*[\(\[]?(?:NASDAQ|NYSE|AMEX|OTC)?:?\s*\1[\)\]]?)?"
        )
        matches = re.findall(ticker_pattern, text)

        for match in matches:
            if match in cls.COMMON_TICKERS:
                return match

        for ticker, company in cls.COMMON_TICKERS.items():
            if company.upper() in text:
                return ticker

        return "GENERAL"


class PromptManager:

    def __init__(self):
        self.prompts = self.get_default_prompts()

    def get_default_prompts(self) -> Dict[str, str]:
        return {
            "sentiment_analysis": """You are a professional financial analyst specializing in market sentiment analysis. Analyze the following news article and provide a detailed sentiment assessment specifically focused on market and stock impact.

ARTICLE DETAILS:
Title: {title}
Source: {source}
Published: {published_at}
Description: {description}
Content: {content}

ANALYSIS REQUIREMENTS:

1. SENTIMENT SCORE: Provide a precise sentiment score from -1.0 to 1.0 where:
   - -1.0 = Extremely negative (bankruptcy, fraud, major lawsuits, severe losses)
   - -0.75 = Very negative (significant losses, downgrades, major problems)
   - -0.5 = Negative (missed earnings, declining sales, minor setbacks)
   - -0.25 = Slightly negative (minor concerns, small headwinds)
   - 0.0 = Neutral (no clear positive or negative impact)
   - 0.25 = Slightly positive (minor improvements, small wins)
   - 0.5 = Positive (beat earnings, growth, upgrades)
   - 0.75 = Very positive (major contracts, breakthrough innovations)
   - 1.0 = Extremely positive (game-changing developments, massive growth)

2. STOCK TICKER: Identify the primary stock ticker if this news is about a specific company. Use "GENERAL" if it's about the overall market or multiple companies.

3. CONFIDENCE LEVEL: Rate your confidence in this analysis from 0.0 to 1.0

4. KEY FACTORS: List the specific factors that influenced your sentiment score

5. MARKET IMPACT: Assess the likely market impact (immediate, short-term, long-term)

Please respond ONLY with a valid JSON object in the following format (no markdown, no code blocks, just JSON):
{{
    "sentiment_score": <float between -1.0 and 1.0>,
    "ticker": "<stock ticker or GENERAL>",
    "confidence": <float between 0.0 and 1.0>,
    "summary": "<2-3 sentence summary of the article>",
    "key_factors": ["<factor1>", "<factor2>", "<factor3>"],
    "market_impact": "<immediate/short-term/long-term impact assessment>",
    "reasoning": "<brief explanation of why this sentiment score was assigned>"
}}

IMPORTANT:
- Be objective and data-driven in your assessment
- Consider both direct and indirect impacts
- Account for market context and sector implications
- Ensure the sentiment score accurately reflects the magnitude of impact
- Return ONLY the JSON object, no additional text or formatting"""
        }

    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        """Get formatted prompt with variables replaced"""
        template = self.prompts.get(prompt_type, self.prompts["sentiment_analysis"])
        return template.format(**kwargs)


class NewsAPIClient:

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"

    def fetch_stock_news(self, ticker: str, from_date: str = None) -> List[NewsArticle]:
        """Fetch news for specific stock ticker"""

        # Map ticker to company name if known
        company_names = {
            "AAPL": "Apple",
            "GOOGL": "Google",
            "MSFT": "Microsoft",
            "AMZN": "Amazon",
            "TSLA": "Tesla",
            "META": "Meta Facebook",
        }

        query = f"{ticker} OR {company_names.get(ticker, ticker)} stock"

        endpoint = f"{self.base_url}/everything"
        params = {
            "apiKey": self.api_key,
            "q": query,
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": 20,
        }

        if from_date:
            params["from"] = from_date
        else:
            week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            params["from"] = week_ago

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get("articles", []):
                article = NewsArticle(
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    source=item.get("source", {}).get("name", "Unknown"),
                    url=item.get("url", ""),
                    published_at=item.get("publishedAt", ""),
                    content=item.get("content", ""),
                    ticker=ticker,
                )
                articles.append(article)

            return articles

        except requests.exceptions.RequestException as e:
            print(f"Error fetching news for {ticker}: {e}")
            return []

    def fetch_market_news(
        self, category: str = "business", page_size: int = 20
    ) -> List[NewsArticle]:
        """Fetch general market news"""

        endpoint = f"{self.base_url}/top-headlines"
        params = {
            "apiKey": self.api_key,
            "category": category,
            "country": "us",
            "pageSize": page_size,
        }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get("articles", []):
                article = NewsArticle(
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    source=item.get("source", {}).get("name", "Unknown"),
                    url=item.get("url", ""),
                    published_at=item.get("publishedAt", ""),
                    content=item.get("content", ""),
                )
                articles.append(article)

            return articles

        except requests.exceptions.RequestException as e:
            print(f"Error fetching market news: {e}")
            return []


class GeminiSentimentAnalyzer:
    """Sentiment analyzer using Google Gemini with -1 to 1 scoring"""

    def __init__(self, api_key: str, prompt_manager: PromptManager):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")
        self.prompt_manager = prompt_manager

    def analyze_article(self, article: NewsArticle) -> Optional[SentimentAnalysis]:
        """Analyze sentiment of a single article"""

        # Get the prompt template
        prompt = self.prompt_manager.get_prompt(
            "sentiment_analysis",
            title=article.title,
            source=article.source,
            published_at=article.published_at,
            description=article.description,
            content=article.content[:1000] if article.content else article.description,
        )

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text

            # Clean up response - remove any markdown formatting
            response_text = response_text.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            # Remove any leading/trailing whitespace
            response_text = response_text.strip()

            # Parse JSON response
            analysis_data = json.loads(response_text)

            # Extract ticker if not already set
            ticker = article.ticker or StockTickerExtractor.extract_ticker(
                article.title, article.content, analysis_data
            )

            # Create sentiment analysis object
            return SentimentAnalysis(
                ticker=ticker,
                news_date=article.published_at[:10]
                if article.published_at
                else datetime.now().strftime("%Y-%m-%d"),
                headline=article.title,
                sentiment_score=float(analysis_data.get("sentiment_score", 0.0)),
                summary=analysis_data.get("summary", ""),
                confidence=float(analysis_data.get("confidence", 0.5)),
                key_topics=analysis_data.get("key_factors", []),
                market_impact=analysis_data.get("market_impact", "unknown"),
                raw_response=analysis_data,
            )

        except Exception as e:
            print(f"Error analyzing article '{article.title[:50]}...': {e}")
            return None


class CSVExporter:
    @staticmethod
    def export_to_csv(analyses: List[SentimentAnalysis], filename: str = None) -> str:
        """Export analyses to CSV file"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_{timestamp}.csv"

        # Prepare CSV data
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "ticker",
                "news_date",
                "headline",
                "sentiment_score",
                "confidence",
                "summary",
                "market_impact",
                "key_topics",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Write data rows
            for analysis in analyses:
                writer.writerow(
                    {
                        "ticker": analysis.ticker,
                        "news_date": analysis.news_date,
                        "headline": analysis.headline,
                        "sentiment_score": round(analysis.sentiment_score, 4),
                        "confidence": round(analysis.confidence, 2),
                        "summary": analysis.summary,
                        "market_impact": analysis.market_impact,
                        "key_topics": "; ".join(analysis.key_topics),
                    }
                )

        print(f"📊 CSV exported to: {filename}")
        return filename

    @staticmethod
    def export_detailed_json(
        analyses: List[SentimentAnalysis], filename: str = None
    ) -> str:

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_details_{timestamp}.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "total_articles": len(analyses),
            "average_sentiment": sum(a.sentiment_score for a in analyses)
            / len(analyses)
            if analyses
            else 0,
            "analyses": [
                {
                    "ticker": a.ticker,
                    "news_date": a.news_date,
                    "headline": a.headline,
                    "sentiment_score": a.sentiment_score,
                    "confidence": a.confidence,
                    "summary": a.summary,
                    "market_impact": a.market_impact,
                    "key_topics": a.key_topics,
                    "full_analysis": a.raw_response,
                }
                for a in analyses
            ],
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"📄 Detailed JSON exported to: {filename}")
        return filename


class StockNewsSentimentAnalyzer:

    def __init__(self, news_api_key: str, gemini_api_key: str):
        self.news_client = NewsAPIClient(news_api_key)
        self.prompt_manager = PromptManager()
        self.sentiment_analyzer = GeminiSentimentAnalyzer(
            gemini_api_key, self.prompt_manager
        )
        self.csv_exporter = CSVExporter()

    def analyze_stocks(
        self, tickers: List[str], max_articles_per_stock: int = 5
    ) -> List[SentimentAnalysis]:

        all_analyses = []

        for ticker in tickers:
            print(f"\n📈 Analyzing {ticker}...")
            articles = self.news_client.fetch_stock_news(ticker)[
                :max_articles_per_stock
            ]

            if not articles:
                print(f"  No articles found for {ticker}")
                continue

            for article in articles:
                print(f"  Analyzing: {article.title[:60]}...")
                analysis = self.sentiment_analyzer.analyze_article(article)
                if analysis:
                    all_analyses.append(analysis)
                    print(
                        f"    Sentiment: {analysis.sentiment_score:+.3f} (Confidence: {analysis.confidence:.1%})"
                    )

                time.sleep(1)  # Rate limiting

        return all_analyses

    def analyze_market(self, max_articles: int = 20) -> List[SentimentAnalysis]:

        print("\n📰 Analyzing market news...")
        articles = self.news_client.fetch_market_news(page_size=max_articles)

        analyses = []
        for i, article in enumerate(articles, 1):
            print(f"  [{i}/{len(articles)}] {article.title[:60]}...")
            analysis = self.sentiment_analyzer.analyze_article(article)
            if analysis:
                analyses.append(analysis)
                print(f"    Sentiment: {analysis.sentiment_score:+.3f}")

            time.sleep(1)  # Rate limiting

        return analyses

    def generate_report(self, analyses: List[SentimentAnalysis]):

        if not analyses:
            print("No analyses to report")
            return

        print("\n" + "=" * 70)
        print("📊 SENTIMENT ANALYSIS REPORT")
        print("=" * 70)

        # Group by ticker
        by_ticker = {}
        for analysis in analyses:
            if analysis.ticker not in by_ticker:
                by_ticker[analysis.ticker] = []
            by_ticker[analysis.ticker].append(analysis)

        # Overall metrics
        avg_sentiment = sum(a.sentiment_score for a in analyses) / len(analyses)
        print(f"\n📈 OVERALL METRICS:")
        print(f"  Articles Analyzed: {len(analyses)}")
        print(f"  Average Sentiment: {avg_sentiment:+.3f}")
        print(f"  Stocks Covered: {len(by_ticker)}")

        # Sentiment distribution
        very_positive = sum(1 for a in analyses if a.sentiment_score > 0.5)
        positive = sum(1 for a in analyses if 0 < a.sentiment_score <= 0.5)
        neutral = sum(1 for a in analyses if -0.1 <= a.sentiment_score <= 0.1)
        negative = sum(1 for a in analyses if -0.5 <= a.sentiment_score < 0)
        very_negative = sum(1 for a in analyses if a.sentiment_score < -0.5)

        print(f"\n📊 SENTIMENT DISTRIBUTION:")
        print(f"  Very Positive (>0.5):  {'█' * very_positive} {very_positive}")
        print(f"  Positive (0 to 0.5):   {'█' * positive} {positive}")
        print(f"  Neutral (-0.1 to 0.1): {'█' * neutral} {neutral}")
        print(f"  Negative (-0.5 to 0):  {'█' * negative} {negative}")
        print(f"  Very Negative (<-0.5): {'█' * very_negative} {very_negative}")

        # Top movers
        print(f"\n🔝 TOP MOVERS:")
        sorted_tickers = sorted(
            by_ticker.items(),
            key=lambda x: sum(a.sentiment_score for a in x[1]) / len(x[1]),
            reverse=True,
        )

        print("  Positive:")
        for ticker, ticker_analyses in sorted_tickers[:3]:
            avg = sum(a.sentiment_score for a in ticker_analyses) / len(ticker_analyses)
            if avg > 0:
                print(f"    {ticker}: {avg:+.3f} ({len(ticker_analyses)} articles)")

        print("  Negative:")
        for ticker, ticker_analyses in sorted_tickers[-3:]:
            avg = sum(a.sentiment_score for a in ticker_analyses) / len(ticker_analyses)
            if avg < 0:
                print(f"    {ticker}: {avg:+.3f} ({len(ticker_analyses)} articles)")

        # Recent headlines
        print(f"\n📰 RECENT HEADLINES:")
        for analysis in sorted(analyses, key=lambda x: x.sentiment_score, reverse=True)[
            :5
        ]:
            emoji = (
                "🟢"
                if analysis.sentiment_score > 0.2
                else "🔴"
                if analysis.sentiment_score < -0.2
                else "🟡"
            )
            print(
                f"  {emoji} [{analysis.ticker}] {analysis.sentiment_score:+.3f}: {analysis.headline[:70]}"
            )

        print("\n" + "=" * 70)


def main():

    print("\n🚀 STOCK NEWS SENTIMENT ANALYZER V2")
    print("=" * 50)

    # Get API keys
    news_api_key = os.getenv("NEWS_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not news_api_key:
        news_api_key = input("Enter your NewsAPI key: ").strip()

    if not gemini_api_key:
        gemini_api_key = input("Enter your Google Gemini API key: ").strip()

    analyzer = StockNewsSentimentAnalyzer(news_api_key, gemini_api_key)

    while True:
        print("\n📋 OPTIONS:")
        print("1. Analyze Specific Stocks")
        print("2. Analyze Market News")
        print("3. Analyze from Watchlist File")
        print("4. Exit")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == "1":
            # Analyze specific stocks
            tickers_input = (
                input("Enter stock tickers (comma-separated, e.g., AAPL,MSFT,GOOGL): ")
                .strip()
                .upper()
            )
            tickers = [t.strip() for t in tickers_input.split(",")]
            max_articles = int(
                input("Max articles per stock [default: 5]: ").strip() or "5"
            )

            analyses = analyzer.analyze_stocks(tickers, max_articles)

            if analyses:
                analyzer.generate_report(analyses)

                # Export options
                print("\n💾 EXPORT OPTIONS:")
                print("1. Export to CSV")
                print("2. Export detailed JSON")
                print("3. Both")
                print("4. Skip")

                export_choice = input("Select (1-4): ").strip()

                if export_choice in ["1", "3"]:
                    analyzer.csv_exporter.export_to_csv(analyses)

                if export_choice in ["2", "3"]:
                    analyzer.csv_exporter.export_detailed_json(analyses)

        elif choice == "2":
            # Analyze market news
            max_articles = int(
                input("Number of articles to analyze [default: 20]: ").strip() or "20"
            )

            analyses = analyzer.analyze_market(max_articles)

            if analyses:
                analyzer.generate_report(analyses)

                # Export options
                print("\n💾 EXPORT OPTIONS:")
                print("1. Export to CSV")
                print("2. Export detailed JSON")
                print("3. Both")
                print("4. Skip")

                export_choice = input("Select (1-4): ").strip()

                if export_choice in ["1", "3"]:
                    analyzer.csv_exporter.export_to_csv(analyses)

                if export_choice in ["2", "3"]:
                    analyzer.csv_exporter.export_detailed_json(analyses)

        elif choice == "3":
            # Load watchlist from file
            watchlist_file = (
                input("Enter watchlist filename [default: watchlist.txt]: ").strip()
                or "watchlist.txt"
            )

            if os.path.exists(watchlist_file):
                with open(watchlist_file, "r") as f:
                    tickers = [line.strip().upper() for line in f if line.strip()]

                print(f"Loaded {len(tickers)} tickers from {watchlist_file}")
                max_articles = int(
                    input("Max articles per stock [default: 3]: ").strip() or "3"
                )

                analyses = analyzer.analyze_stocks(tickers, max_articles)

                if analyses:
                    analyzer.generate_report(analyses)
                    analyzer.csv_exporter.export_to_csv(analyses)
                    analyzer.csv_exporter.export_detailed_json(analyses)
            else:
                print(f"❌ File {watchlist_file} not found")

        elif choice == "4":
            print("\n👋 Goodbye!")
            break

        else:
            print("❌ Invalid option")


if __name__ == "__main__":
    main()
