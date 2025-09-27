import requests
from bs4 import BeautifulSoup
import time
import random
import re
from urllib.parse import quote_plus, urlparse
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import Counter
import hashlib

class PlagiarismChecker:
    """Fast and efficient plagiarism detection system"""
    
    def __init__(self):
        # Initialize session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Download NLTK data if not already present
        try:
            nltk.data.find('tokenizers/punkt_tab')
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        
        # Enhanced parameters for better detection
        self.chunk_size = 15  # Smaller chunks for better granularity
        self.max_chunks = 20  # More chunks for thorough analysis
        self.similarity_threshold = 0.15  # Much lower threshold for exact matches
        self.exact_match_threshold = 0.85  # High threshold for exact matches
        self.timeout = 5  # Longer timeout for better content fetching
        self.max_workers = 6  # More parallel processing
        
        # Multi-layer detection settings
        self.min_exact_words = 6  # Minimum consecutive words for exact match
        self.ngram_sizes = [3, 4, 5]  # N-gram sizes for analysis
        self.sentence_overlap_threshold = 0.7  # Sentence similarity threshold
        
        print("Fast Plagiarism Checker initialized")
    
    def check_plagiarism(self, text):
        """
        Fast plagiarism analysis with optimized processing
        """
        try:
            print(f"Starting fast plagiarism analysis for {len(text)} characters...")
            
            # Preprocess text
            clean_text = self._preprocess_text(text)
            
            # Create optimized chunks
            chunks = self._create_optimized_chunks(clean_text)
            
            if not chunks:
                return self._create_minimal_report(text, "Text too short for analysis")
            
            print(f"Processing {len(chunks)} optimized chunks in parallel...")
            
            # Process chunks in parallel
            all_matches = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all chunk searches simultaneously
                future_to_chunk = {
                    executor.submit(self._search_chunk_fast, chunk, idx): (chunk, idx) 
                    for idx, chunk in enumerate(chunks)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    chunk, idx = future_to_chunk[future]
                    try:
                        matches = future.result()
                        if matches:
                            all_matches.extend(matches)
                            print(f"Chunk {idx + 1}/{len(chunks)}: Found {len(matches)} matches")
                        else:
                            print(f"Chunk {idx + 1}/{len(chunks)}: No matches")
                    except Exception as e:
                        print(f"Chunk {idx + 1}/{len(chunks)}: Error - {str(e)}")
            
            # Calculate results
            total_words = len(clean_text.split())
            plagiarized_words = self._calculate_plagiarized_words(all_matches, clean_text)
            plagiarism_percentage = (plagiarized_words / total_words) * 100 if total_words > 0 else 0
            
            # Group sources
            sources = self._group_sources(all_matches)
            
            print(f"Analysis complete: {plagiarism_percentage:.2f}% plagiarism detected")
            
            return {
                'plagiarism_percentage': round(plagiarism_percentage, 2),
                'total_words': total_words,
                'plagiarized_words': plagiarized_words,
                'sources_found': len(sources),
                'matches': all_matches[:15],  # Top matches
                'sources': sources[:8],  # Top sources
                'analysis_summary': self._generate_summary(plagiarism_percentage, len(sources)),
                'diagnostic_info': {
                    'chunks_processed': len(chunks),
                    'successful_matches': len(all_matches),
                    'processing_time': 'optimized'
                }
            }
            
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return self._create_error_report(str(e))
    
    def _preprocess_text(self, text):
        """Clean and prepare text for analysis"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove citations and references
        text = re.sub(r'\[[0-9]+\]', '', text)
        text = re.sub(r'\([^)]*\d{4}[^)]*\)', '', text)
        text = re.sub(r'http[s]?://[^\s]+', '', text)
        
        return text
    
    def _create_optimized_chunks(self, text):
        """Create smaller, overlapping chunks for thorough analysis"""
        words = text.split()
        chunks = []
        
        # Create overlapping chunks with better coverage
        chunk_overlap = 8  # More overlap for better detection
        step_size = max(1, self.chunk_size - chunk_overlap)
        
        for i in range(0, len(words), step_size):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) >= 6:  # Lower minimum for better detection
                chunk = ' '.join(chunk_words)
                chunks.append(chunk)
                
                # Limit chunks but allow more for thorough analysis
                if len(chunks) >= self.max_chunks:
                    break
        
        # Also add sentence-based chunks for better detection
        sentences = sent_tokenize(text)
        for sentence in sentences[:5]:  # Add top sentences as chunks
            if len(sentence.split()) >= 6 and len(chunks) < self.max_chunks:
                chunks.append(sentence)
        
        return chunks
    
    def _search_chunk_fast(self, chunk, chunk_idx):
        """Fast search for a text chunk using multiple engines"""
        try:
            # Add small delay for rate limiting
            time.sleep(random.uniform(0.5, 1.0))
            
            print(f"Searching chunk {chunk_idx + 1}: {chunk[:50]}...")
            
            # Try multiple search engines quickly
            search_results = []
            
            # Try DuckDuckGo first (usually more permissive)
            try:
                duckduckgo_results = self._search_duckduckgo_fast(chunk)
                search_results.extend(duckduckgo_results)
            except:
                pass
            
            # If no results, try Bing
            if not search_results:
                try:
                    bing_results = self._search_bing_fast(chunk)
                    search_results.extend(bing_results)
                except:
                    pass
            
            # Process found URLs
            matches = []
            for result in search_results[:4]:  # Limit to top 4 results
                try:
                    content = self._fetch_content_fast(result['url'])
                    if content:
                        similarity = self._calculate_similarity_fast(chunk, content)
                        # Use lower threshold but prioritize high-confidence matches
                        if similarity > self.similarity_threshold or similarity > self.exact_match_threshold:
                            # Find the actual matching substring in the content
                            matching_portion = self._find_matching_portion(chunk, content)
                            matches.append({
                                'original_text': chunk,
                                'matched_text': matching_portion[:200] + '...' if len(matching_portion) > 200 else matching_portion,
                                'full_matched_content': matching_portion,  # Store full match for calculation
                                'url': result['url'],
                                'title': result.get('title', 'Untitled'),
                                'similarity': similarity,
                                'chunk_index': chunk_idx,
                                'chunk_word_count': len(chunk.split())  # Track original chunk size
                            })
                except Exception as e:
                    continue
            
            return matches
            
        except Exception as e:
            print(f"Error in chunk {chunk_idx}: {str(e)}")
            return []
    
    def _search_duckduckgo_fast(self, query):
        """Fast DuckDuckGo search"""
        try:
            # Create search query
            search_query = f'"{query[:100]}"'  # Limit query length
            search_url = f"https://duckduckgo.com/html/?q={quote_plus(search_query)}"
            
            response = self.session.get(search_url, timeout=self.timeout)
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Parse results
            for result_link in soup.find_all('a', class_='result__a')[:6]:
                try:
                    if result_link and hasattr(result_link, 'get_text') and hasattr(result_link, 'get'):
                        title = result_link.get_text(strip=True)
                        href_attr = result_link.get('href')
                        url = href_attr if href_attr else ''
                        
                        if url and self._is_valid_url(url):
                            results.append({'title': title, 'url': url})
                except:
                    continue
            
            return results
            
        except Exception as e:
            print(f"DuckDuckGo search failed: {str(e)}")
            return []
    
    def _search_bing_fast(self, query):
        """Fast Bing search as fallback"""
        try:
            search_query = f'"{query[:100]}"'
            search_url = f"https://www.bing.com/search?q={quote_plus(search_query)}"
            
            response = self.session.get(search_url, timeout=self.timeout)
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Parse Bing results
            for result in soup.find_all('h2')[:6]:
                try:
                    if result and hasattr(result, 'find'):
                        link_element = result.find('a')
                        if link_element and hasattr(link_element, 'get') and hasattr(link_element, 'get_text'):
                            href_attr = link_element.get('href')
                            if href_attr:
                                title = link_element.get_text(strip=True)
                                url = str(href_attr)
                                
                                if url and self._is_valid_url(url):
                                    results.append({'title': title, 'url': url})
                except:
                    continue
            
            return results
            
        except Exception as e:
            print(f"Bing search failed: {str(e)}")
            return []
    
    def _fetch_content_fast(self, url):
        """Fast content fetching with timeout"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove unwanted elements
                for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                    tag.decompose()
                
                # Get text content
                text = soup.get_text(separator=' ', strip=True)
                text = re.sub(r'\s+', ' ', text)
                
                return text if len(text) > 100 else None
        except:
            return None
    
    def _calculate_similarity_fast(self, text1, text2):
        """Multi-layer similarity calculation with exact match detection"""
        try:
            text1_clean = text1.lower().strip()
            text2_clean = text2.lower().strip()
            
            # Layer 1: Exact substring matching
            exact_match_score = self._calculate_exact_match(text1_clean, text2_clean)
            if exact_match_score > 0.8:
                return exact_match_score
            
            # Layer 2: Sequence similarity (SequenceMatcher)
            seq_similarity = SequenceMatcher(None, text1_clean, text2_clean).ratio()
            
            # Layer 3: Word overlap with position weighting
            word_overlap = self._calculate_weighted_word_overlap(text1_clean, text2_clean)
            
            # Layer 4: N-gram similarity
            ngram_similarity = self._calculate_ngram_similarity(text1_clean, text2_clean)
            
            # Layer 5: TF-IDF cosine similarity
            tfidf_similarity = self._calculate_tfidf_similarity(text1_clean, text2_clean)
            
            # Layer 6: Sentence-level comparison
            sentence_similarity = self._calculate_sentence_similarity(text1, text2)
            
            # Weighted combination of all methods
            final_similarity = (
                exact_match_score * 0.3 +
                seq_similarity * 0.2 +
                word_overlap * 0.15 +
                ngram_similarity * 0.15 +
                tfidf_similarity * 0.1 +
                sentence_similarity * 0.1
            )
            
            return min(final_similarity, 1.0)
            
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return 0
    
    def _calculate_exact_match(self, text1, text2):
        """Calculate exact substring matching score"""
        try:
            # Find exact matches of consecutive words
            words1 = text1.split()
            words2 = text2.split()
            
            if not words1 or not words2:
                return 0
            
            max_match_length = 0
            total_matches = 0
            
            # Check for consecutive word matches
            for i in range(len(words1) - self.min_exact_words + 1):
                for j in range(len(words2) - self.min_exact_words + 1):
                    match_length = 0
                    while (i + match_length < len(words1) and 
                           j + match_length < len(words2) and
                           words1[i + match_length] == words2[j + match_length]):
                        match_length += 1
                    
                    if match_length >= self.min_exact_words:
                        max_match_length = max(max_match_length, match_length)
                        total_matches += match_length
            
            # Calculate score based on exact matches
            if total_matches > 0:
                return min(total_matches / len(words1), 1.0)
            return 0
            
        except:
            return 0
    
    def _calculate_weighted_word_overlap(self, text1, text2):
        """Calculate word overlap with position weighting"""
        try:
            words1 = word_tokenize(text1)
            words2 = word_tokenize(text2)
            
            if not words1 or not words2:
                return 0
            
            # Remove stopwords for better matching
            words1_filtered = [w for w in words1 if w not in self.stop_words and len(w) > 2]
            words2_filtered = [w for w in words2 if w not in self.stop_words and len(w) > 2]
            
            if not words1_filtered or not words2_filtered:
                return len(set(words1).intersection(set(words2))) / len(set(words1).union(set(words2)))
            
            # Calculate intersection and union
            intersection = len(set(words1_filtered).intersection(set(words2_filtered)))
            union = len(set(words1_filtered).union(set(words2_filtered)))
            
            return intersection / union if union > 0 else 0
            
        except:
            return 0
    
    def _calculate_ngram_similarity(self, text1, text2):
        """Calculate n-gram similarity across multiple n-gram sizes"""
        try:
            total_similarity = 0
            count = 0
            
            for n in self.ngram_sizes:
                # Generate n-grams
                words1 = word_tokenize(text1)
                words2 = word_tokenize(text2)
                
                if len(words1) >= n and len(words2) >= n:
                    ngrams1 = set(ngrams(words1, n))
                    ngrams2 = set(ngrams(words2, n))
                    
                    if ngrams1 and ngrams2:
                        intersection = len(ngrams1.intersection(ngrams2))
                        union = len(ngrams1.union(ngrams2))
                        similarity = intersection / union if union > 0 else 0
                        total_similarity += similarity
                        count += 1
            
            return total_similarity / count if count > 0 else 0
            
        except:
            return 0
    
    def _calculate_tfidf_similarity(self, text1, text2):
        """Calculate TF-IDF cosine similarity"""
        try:
            if not text1.strip() or not text2.strip():
                return 0
            
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=1000,
                lowercase=True
            )
            
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity
            
        except:
            return 0
    
    def _calculate_sentence_similarity(self, text1, text2):
        """Calculate sentence-level similarity"""
        try:
            sentences1 = sent_tokenize(text1)
            sentences2 = sent_tokenize(text2)
            
            if not sentences1 or not sentences2:
                return 0
            
            max_similarities = []
            
            for sent1 in sentences1:
                sent_similarities = []
                for sent2 in sentences2:
                    # Calculate similarity between sentences
                    sim = SequenceMatcher(None, sent1.lower(), sent2.lower()).ratio()
                    sent_similarities.append(sim)
                
                if sent_similarities:
                    max_similarities.append(max(sent_similarities))
            
            # Average of maximum similarities
            return sum(max_similarities) / len(max_similarities) if max_similarities else 0
            
        except:
            return 0

    def _is_valid_url(self, url):
        """Check if URL is valid for analysis"""
        if not url or not url.startswith(('http://', 'https://')):
            return False
        
        try:
            domain = urlparse(url).netloc.lower()
            excluded = ['google.', 'youtube.', 'facebook.', 'twitter.', 'instagram.']
            return not any(ex in domain for ex in excluded)
        except:
            return False
    
    def _find_matching_portion(self, chunk, content):
        """Find the best matching portion in content for the given chunk"""
        try:
            chunk_words = chunk.lower().split()
            content_words = content.lower().split()
            
            if not chunk_words or not content_words:
                return content[:500]  # Return reasonable portion
            
            # Find the best matching sequence
            best_match_start = 0
            best_match_length = 0
            best_match_score = 0
            
            # Look for consecutive word matches
            for i in range(len(content_words) - len(chunk_words) + 1):
                score = 0
                for j in range(min(len(chunk_words), len(content_words) - i)):
                    if j < len(chunk_words) and chunk_words[j] == content_words[i + j]:
                        score += 1
                    else:
                        break
                
                if score > best_match_score:
                    best_match_score = score
                    best_match_start = i
                    best_match_length = score
            
            # Extract the matching portion with some context
            start_idx = max(0, best_match_start - 10)
            end_idx = min(len(content_words), best_match_start + best_match_length + 10)
            matching_words = content_words[start_idx:end_idx]
            
            return ' '.join(matching_words)
            
        except:
            return content[:500]  # Fallback to content portion
    
    def _calculate_plagiarized_words(self, matches, original_text):
        """Calculate plagiarized word count based on actual matches"""
        if not matches:
            return 0
        
        total_words = len(original_text.split())
        
        # Calculate based on high-confidence matches
        high_confidence_matches = [m for m in matches if m['similarity'] > 0.7]
        if high_confidence_matches:
            # Use actual chunk word counts for high-confidence matches
            plagiarized_words = sum(m.get('chunk_word_count', 0) for m in high_confidence_matches)
            return min(plagiarized_words, total_words)
        
        # Fallback: use similarity-weighted calculation
        total_similarity_score = sum(m['similarity'] * len(m['original_text'].split()) for m in matches)
        estimated_plagiarized_words = int(total_similarity_score * 0.8)  # Conservative estimate
        
        return min(estimated_plagiarized_words, total_words)
    
    def _group_sources(self, matches):
        """Group matches by source"""
        sources = {}
        
        for match in matches:
            url = match['url']
            if url not in sources:
                sources[url] = {
                    'url': url,
                    'title': match['title'],
                    'matches': [],
                    'avg_similarity': 0,
                    'match_count': 0
                }
            
            sources[url]['matches'].append(match)
            sources[url]['match_count'] += 1
        
        # Calculate averages
        for source in sources.values():
            similarities = [m['similarity'] for m in source['matches']]
            source['avg_similarity'] = sum(similarities) / len(similarities)
        
        # Sort by similarity
        return sorted(sources.values(), key=lambda x: x['avg_similarity'], reverse=True)
    
    def _generate_summary(self, percentage, source_count):
        """Generate analysis summary"""
        if percentage == 0:
            return "No plagiarism detected. The text appears to be original."
        elif percentage < 15:
            return f"Low plagiarism detected ({percentage}%). Found {source_count} potential sources."
        elif percentage < 30:
            return f"Moderate plagiarism detected ({percentage}%). Review recommended. {source_count} sources found."
        else:
            return f"High plagiarism detected ({percentage}%). Revision needed. {source_count} sources found."
    
    def _create_minimal_report(self, text, message):
        """Create minimal report"""
        return {
            'plagiarism_percentage': 0,
            'total_words': len(text.split()),
            'plagiarized_words': 0,
            'sources_found': 0,
            'matches': [],
            'sources': [],
            'analysis_summary': message
        }
    
    def _create_error_report(self, error_message):
        """Create error report"""
        return {
            'error': f"Analysis failed: {error_message}",
            'plagiarism_percentage': 0,
            'total_words': 0,
            'plagiarized_words': 0,
            'sources_found': 0,
            'matches': [],
            'sources': []
        }