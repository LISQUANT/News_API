import csv
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from google import genai
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

    ticker: str
    news_date: str
    headline: str
    sentiment_score: float  # -1 to 1
    summary: str
    confidence: float
    key_topics: List[str]
    market_impact: str
    raw_response: Dict[str, Any]


@dataclass
class StockAggregateAnalysis:

    ticker: str
    num_articles: int
    weighted_avg_score: float  # (sentiment_score * confidence) aggregated
    ai_summary: str
    date_range: str
    individual_analyses: List[SentimentAnalysis]


class StockTickerExtractor:

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

    def __init__(self, prompts_file: str = "prompts.json"):
        self.prompts_file = prompts_file
        self.prompts = self.load_prompts()

    def load_prompts(self) -> Dict[str, str]:
        """Load prompts from JSON file"""
        if os.path.exists(self.prompts_file):
            with open(self.prompts_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Error: {self.prompts_file} not found. Please ensure the prompts.json file exists.")

    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        template = self.prompts.get(prompt_type, self.prompts.get("sentiment_analysis", ""))
        return template.format(**kwargs)


class NewsAPIClient:

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"

    def fetch_stock_news(self, ticker: str, from_date: str = None) -> List[NewsArticle]:

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
        #Fetch general market news

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
    #Sentiment analyzer using Google Gemini with -1 to 1 scoring

    def __init__(self, api_key: str, prompt_manager: PromptManager):
        # Initialize the Gemini client - matches Google documentation
        self.client = genai.Client(api_key=api_key)
        self.prompt_manager = prompt_manager

    def analyze_article(self, article: NewsArticle) -> Optional[SentimentAnalysis]:
        prompt = self.prompt_manager.get_prompt(
            "sentiment_analysis",
            title=article.title,
            source=article.source,
            published_at=article.published_at,
            description=article.description,
            content=article.content[:1000] if article.content else article.description,
        )

        try:
            # Use exact syntax from Google documentation
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
            )
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

    def generate_batch_summary(
        self, ticker: str, analyses: List[SentimentAnalysis]
    ) -> str:
        """Generate an AI summary for a batch of news articles about a stock"""

        if not analyses:
            return "No articles analyzed"

        articles_summary = "\n\n".join(
            [
                f"Article {i+1}:\n"
                f"  Headline: {a.headline}\n"
                f"  Sentiment Score: {a.sentiment_score:+.2f}\n"
                f"  Confidence: {a.confidence:.0%}\n"
                f"  Summary: {a.summary}"
                for i, a in enumerate(analyses)
            ]
        )

        prompt = self.prompt_manager.get_prompt(
            "batch_summary",
            ticker=ticker,
            num_articles=len(analyses),
            articles_summary=articles_summary,
        )

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
            )
            return response.text.strip()

        except Exception as e:
            print(f"Error generating batch summary for {ticker}: {e}")
            return f"Error generating summary: {str(e)}"


class CSVExporter:
    @staticmethod
    def export_aggregates_to_csv(
        aggregates: List[StockAggregateAnalysis],
        filename: str = "sentiment_analysis.csv",
    ) -> str:

        file_exists = os.path.exists(filename)

        with open(filename, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "ticker",
                "num_articles",
                "weighted_avg_score",
                "date_range",
                "ai_summary",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for agg in aggregates:
                writer.writerow(
                    {
                        "ticker": agg.ticker,
                        "num_articles": agg.num_articles,
                        "weighted_avg_score": round(agg.weighted_avg_score, 4),
                        "date_range": agg.date_range,
                        "ai_summary": agg.ai_summary,
                    }
                )

        action = "appended to" if file_exists else "created"
        print(f"{len(aggregates)} stock aggregates {action}: {filename}")
        return filename

    @staticmethod
    def export_to_csv(
        analyses: List[SentimentAnalysis], filename: str = "sentiment_analysis.csv"
    ) -> str:
        """Export analyses to CSV file, appending if file exists"""

        file_exists = os.path.exists(filename)

        with open(filename, "a", newline="", encoding="utf-8") as csvfile:
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

            if not file_exists:
                writer.writeheader()

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

        action = "appended to" if file_exists else "created"
        print(f"{len(analyses)} analyses {action}: {filename}")
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

        print(f"Detailed JSON exported to: {filename}")
        return filename


class StockNewsSentimentAnalyzer:

    def __init__(self, news_api_key: str, gemini_api_key: str):
        self.news_client = NewsAPIClient(news_api_key)
        self.prompt_manager = PromptManager()
        self.sentiment_analyzer = GeminiSentimentAnalyzer(
            gemini_api_key, self.prompt_manager
        )
        self.csv_exporter = CSVExporter()

    def aggregate_stock_analysis(
        self, ticker: str, analyses: List[SentimentAnalysis]
    ) -> StockAggregateAnalysis:

        if not analyses:
            return StockAggregateAnalysis(
                ticker=ticker,
                num_articles=0,
                weighted_avg_score=0.0,
                ai_summary="No articles analyzed",
                date_range="N/A",
                individual_analyses=[],
            )

        total_weighted_score = sum(
            a.sentiment_score * a.confidence for a in analyses
        )
        total_confidence = sum(a.confidence for a in analyses)

        weighted_avg = (
            total_weighted_score / total_confidence if total_confidence > 0 else 0.0
        )

        dates = [a.news_date for a in analyses if a.news_date]
        date_range = (
            f"{min(dates)} to {max(dates)}" if dates and len(set(dates)) > 1 else dates[0] if dates else "N/A"
        )

        print(f"  Generating AI summary for {ticker}...")
        ai_summary = self.sentiment_analyzer.generate_batch_summary(ticker, analyses)

        return StockAggregateAnalysis(
            ticker=ticker,
            num_articles=len(analyses),
            weighted_avg_score=weighted_avg,
            ai_summary=ai_summary,
            date_range=date_range,
            individual_analyses=analyses,
        )

    def analyze_stocks(
        self, tickers: List[str], max_articles_per_stock: int = 5
    ) -> List[StockAggregateAnalysis]:

        stock_aggregates = []

        for ticker in tickers:
            print(f"\nAnalyzing {ticker}...")
            articles = self.news_client.fetch_stock_news(ticker)[
                :max_articles_per_stock
            ]

            if not articles:
                print(f"  No articles found for {ticker}")
                continue

            ticker_analyses = []
            for article in articles:
                print(f"  Analyzing: {article.title[:60]}...")
                analysis = self.sentiment_analyzer.analyze_article(article)
                if analysis:
                    ticker_analyses.append(analysis)
                    print(
                        f"    Sentiment: {analysis.sentiment_score:+.3f} (Confidence: {analysis.confidence:.1%})"
                    )

                time.sleep(1)  # Rate limiting

            if ticker_analyses:
                aggregate = self.aggregate_stock_analysis(ticker, ticker_analyses)
                stock_aggregates.append(aggregate)
                print(
                    f"  {ticker} Weighted Average: {aggregate.weighted_avg_score:+.3f} ({aggregate.num_articles} articles)"
                )

        return stock_aggregates

    def analyze_market(self, max_articles: int = 20) -> List[SentimentAnalysis]:

        print("\nAnalyzing market news...")
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

    def generate_report(self, aggregates: List[StockAggregateAnalysis]):

        if not aggregates:
            print("No analyses to report")
            return

        print("\n" + "=" * 80)
        print("STOCK SENTIMENT ANALYSIS REPORT (AGGREGATED)")
        print("=" * 80)

        # Overall metrics
        total_articles = sum(agg.num_articles for agg in aggregates)
        avg_weighted_score = (
            sum(agg.weighted_avg_score for agg in aggregates) / len(aggregates)
            if aggregates
            else 0
        )

        print(f"\nOVERALL METRICS:")
        print(f"  Stocks Analyzed: {len(aggregates)}")
        print(f"  Total Articles: {total_articles}")
        print(f"  Average Weighted Score: {avg_weighted_score:+.4f}")

        # Sentiment distribution
        very_positive = sum(1 for a in aggregates if a.weighted_avg_score > 0.5)
        positive = sum(1 for a in aggregates if 0 < a.weighted_avg_score <= 0.5)
        neutral = sum(
            1 for a in aggregates if -0.1 <= a.weighted_avg_score <= 0.1
        )
        negative = sum(
            1 for a in aggregates if -0.5 <= a.weighted_avg_score < 0
        )
        very_negative = sum(1 for a in aggregates if a.weighted_avg_score < -0.5)

        print(f"\nSENTIMENT DISTRIBUTION:")
        print(f"  Very Positive (>0.5):  {'*' * very_positive} {very_positive}")
        print(f"  Positive (0 to 0.5):   {'*' * positive} {positive}")
        print(f"  Neutral (-0.1 to 0.1): {'*' * neutral} {neutral}")
        print(f"  Negative (-0.5 to 0):  {'*' * negative} {negative}")
        print(f"  Very Negative (<-0.5): {'*' * very_negative} {very_negative}")

        # Stock rankings
        print(f"\nSTOCK RANKINGS BY WEIGHTED SENTIMENT:")
        sorted_aggregates = sorted(
            aggregates, key=lambda x: x.weighted_avg_score, reverse=True
        )

        for i, agg in enumerate(sorted_aggregates, 1):
            indicator = (
                "[++]"
                if agg.weighted_avg_score > 0.5
                else "[+]"
                if agg.weighted_avg_score > 0.2
                else "[=]"
                if agg.weighted_avg_score > -0.2
                else "[-]"
                if agg.weighted_avg_score > -0.5
                else "[--]"
            )
            print(
                f"  {i}. {indicator} {agg.ticker}: {agg.weighted_avg_score:+.4f} "
                f"({agg.num_articles} articles)"
            )

        # Detailed summaries
        print(f"\n{'=' * 80}")
        print("AI-GENERATED SUMMARIES BY STOCK")
        print("=" * 80)

        for agg in sorted_aggregates:
            print(f"\n{agg.ticker} - Weighted Score: {agg.weighted_avg_score:+.4f}")
            print(f"Date Range: {agg.date_range}")
            print(f"Articles Analyzed: {agg.num_articles}")
            print(f"\nSummary:")
            print(f"  {agg.ai_summary}")
            print("-" * 80)

        print("\n" + "=" * 80)


def main():

    print("\nSTOCK NEWS SENTIMENT ANALYZER V2")
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
        print("\nOPTIONS:")
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

            aggregates = analyzer.analyze_stocks(tickers, max_articles)

            if aggregates:
                analyzer.generate_report(aggregates)
                # Automatically export to CSV
                analyzer.csv_exporter.export_aggregates_to_csv(aggregates)

        elif choice == "2":
            # Analyze market news
            max_articles = int(
                input("Number of articles to analyze [default: 20]: ").strip() or "20"
            )

            analyses = analyzer.analyze_market(max_articles)

            if analyses:
                analyzer.generate_report(analyses)
                # Automatically export to CSV
                analyzer.csv_exporter.export_to_csv(analyses)

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

                aggregates = analyzer.analyze_stocks(tickers, max_articles)

                if aggregates:
                    analyzer.generate_report(aggregates)
                    # Automatically export to CSV
                    analyzer.csv_exporter.export_aggregates_to_csv(aggregates)
            else:
                print(f"ERROR: File {watchlist_file} not found")

        elif choice == "4":
            print("\nGoodbye!")
            break

        else:
            print("ERROR: Invalid option")


if __name__ == "__main__":
    main()
