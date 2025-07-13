import sqlite3
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import uuid
from datetime import datetime
import re
import pickle
import os
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Try to import sentence-transformers, fall back to TF-IDF if not available
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedRAGSystem:
    def __init__(self, db_path: str = "rag_database.db", model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model_name = model_name
        self.embedding_model = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
        self.property_cache = {}  # Cache for faster property lookups
        
        self.init_database()
        self.load_models()
        self.build_property_cache()
    
    def load_models(self):
        """Load embedding model or TF-IDF vectorizer"""
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {str(e)}")
        
        # Fallback to TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        logger.info("Using TF-IDF vectorizer for document retrieval")
    
    def init_database(self):
        """Initialize RAG database with enhanced property table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced documents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    content TEXT,
                    metadata TEXT,
                    embeddings BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source_file TEXT,
                    doc_type TEXT,
                    searchable_content TEXT
                )
            ''')
            
            # Enhanced property listings table with better indexing
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS property_listings (
                    unique_id INTEGER PRIMARY KEY,
                    property_address TEXT NOT NULL,
                    floor TEXT,
                    suite TEXT,
                    size_sf INTEGER,
                    rent_sf_year TEXT,
                    associate_1 TEXT,
                    broker_email TEXT,
                    associate_2 TEXT,
                    associate_3 TEXT,
                    associate_4 TEXT,
                    annual_rent TEXT,
                    monthly_rent TEXT,
                    gci_on_3_years TEXT,
                    embeddings BLOB,
                    searchable_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add user preferences tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_property_interactions (
                    interaction_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    property_id INTEGER,
                    interaction_type TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    details TEXT,
                    FOREIGN KEY (property_id) REFERENCES property_listings (unique_id)
                )
            ''')
            
            # Create comprehensive indexes for faster searches
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(doc_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_property_address ON property_listings(property_address)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_size_sf ON property_listings(size_sf)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_searchable_text ON property_listings(searchable_text)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_interactions ON user_property_interactions(user_id)')
            
            conn.commit()
            conn.close()
            logger.info("Enhanced RAG database initialized successfully")
            
        except Exception as e:
            logger.error(f"RAG database initialization error: {str(e)}")
            raise
    
    def build_property_cache(self):
        """Build cache for faster property lookups"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT unique_id, property_address, size_sf, floor, suite FROM property_listings')
            rows = cursor.fetchall()
            
            for row in rows:
                self.property_cache[row[0]] = {
                    'address': row[1],
                    'size': row[2],
                    'floor': row[3],
                    'suite': row[4]
                }
            
            conn.close()
            logger.info(f"Built property cache with {len(self.property_cache)} properties")
            
        except Exception as e:
            logger.error(f"Property cache build error: {str(e)}")
    
    def generate_embeddings(self, text: str) -> Optional[List[float]]:
        """Generate embeddings for text"""
        try:
            if self.embedding_model:
                embeddings = self.embedding_model.encode([text])[0]
                return embeddings.tolist()
            return None
        except Exception as e:
            logger.error(f"Embedding generation error: {str(e)}")
            return None
    
    async def load_csv_data(self, file_path: str):
        """Enhanced CSV loading with better property handling"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loading CSV with {len(df)} rows")
            
            # Clean column names - handle various formats
            df.columns = df.columns.str.strip()
            column_mapping = {
                'Property Address': 'property_address',
                'Floor': 'floor', 
                'Suite': 'suite',
                'Size (SF)': 'size_sf',
                'Rent/SF/Year': 'rent_sf_year',
                'Associate 1': 'associate_1',
                'BROKER Email ID': 'broker_email',
                'Associate 2': 'associate_2',
                'Associate 3': 'associate_3', 
                'Associate 4': 'associate_4',
                'Annual Rent': 'annual_rent',
                'Monthly Rent': 'monthly_rent',
                'GCI On 3 Years': 'gci_on_3_years',
                'unique_id': 'unique_id'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Check if this is property data
            if 'property_address' in df.columns or 'unique_id' in df.columns:
                await self.load_enhanced_property_data(df, file_path)
            else:
                await self.load_generic_csv(df, file_path)
            
            # Rebuild caches
            self.build_property_cache()
            if self.tfidf_vectorizer:
                self.rebuild_tfidf_matrix()
                
        except Exception as e:
            logger.error(f"CSV loading error: {str(e)}")
            raise
    
    async def load_enhanced_property_data(self, df: pd.DataFrame, source_file: str):
        """Enhanced property data loading with better search indexing"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for _, row in df.iterrows():
                # Create comprehensive searchable content
                searchable_parts = []
                
                # Add all relevant fields to searchable content
                for col, val in row.items():
                    if pd.notna(val) and val != '' and str(val).lower() not in ['nan', 'none']:
                        if col == 'size_sf':
                            # Add multiple size formats for better matching
                            searchable_parts.append(f"size: {val} sf")
                            searchable_parts.append(f"size: {val} square feet")
                            searchable_parts.append(f"{val} sq ft")
                            searchable_parts.append(f"{val} square feet")
                        else:
                            searchable_parts.append(f"{col.replace('_', ' ')}: {val}")
                
                searchable_text = " | ".join(searchable_parts)
                
                # Create content for embeddings
                content = searchable_text
                embeddings = self.generate_embeddings(content)
                
                # Clean and validate data
                size_sf = None
                if pd.notna(row.get('size_sf')):
                    try:
                        size_sf = int(float(str(row.get('size_sf')).replace(',', '')))
                    except (ValueError, TypeError):
                        pass
                
                # Store in property_listings table with enhanced data
                cursor.execute('''
                    INSERT OR REPLACE INTO property_listings 
                    (unique_id, property_address, floor, suite, size_sf, rent_sf_year, 
                     associate_1, broker_email, associate_2, associate_3, associate_4, 
                     annual_rent, monthly_rent, gci_on_3_years, embeddings, searchable_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row.get('unique_id'),
                    row.get('property_address'),
                    row.get('floor'),
                    row.get('suite'),
                    size_sf,
                    row.get('rent_sf_year'),
                    row.get('associate_1'),
                    row.get('broker_email'),
                    row.get('associate_2'),
                    row.get('associate_3'),
                    row.get('associate_4'),
                    row.get('annual_rent'),
                    row.get('monthly_rent'),
                    row.get('gci_on_3_years'),
                    pickle.dumps(embeddings) if embeddings else None,
                    searchable_text
                ))
                
                # Also store as generic document
                doc_id = str(uuid.uuid4())
                cursor.execute('''
                    INSERT INTO documents 
                    (doc_id, content, metadata, embeddings, source_file, doc_type, searchable_content)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    doc_id,
                    content,
                    json.dumps(row.to_dict()),
                    pickle.dumps(embeddings) if embeddings else None,
                    source_file,
                    'property_listing',
                    searchable_text
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Enhanced loading of {len(df)} property listings completed")
            
        except Exception as e:
            logger.error(f"Enhanced property data loading error: {str(e)}")
            raise
    
    async def get_relevant_context(self, query: str, top_k: int = 10, user_id: str = None) -> List[Dict[str, Any]]:
        """Enhanced context retrieval with exact matching capabilities"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for exact size match first
            exact_results = await self.search_exact_property_match(query, cursor)
            if exact_results:
                logger.info(f"Found {len(exact_results)} exact matches for query: {query}")
                conn.close()
                return exact_results
            
            # Enhanced property search with multiple strategies
            property_keywords = ['property', 'office', 'space', 'sq ft', 'square feet', 'sf', 'rent', 'lease', 'building']
            is_property_search = any(keyword in query.lower() for keyword in property_keywords)
            
            if is_property_search:
                results = await self.enhanced_property_search(query, cursor, top_k=50, user_id=user_id)
            else:
                # Regular search for non-property queries
                if self.embedding_model:
                    query_embeddings = self.generate_embeddings(query)
                    if query_embeddings:
                        results = await self.semantic_search(query_embeddings, cursor, top_k)
                    else:
                        results = await self.enhanced_keyword_search(query, cursor, top_k)
                else:
                    results = await self.enhanced_keyword_search(query, cursor, top_k)
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Enhanced get relevant context error: {str(e)}")
            return []
    
    async def search_exact_property_match(self, query: str, cursor) -> List[Dict[str, Any]]:
        """Search for exact property matches (e.g., exact size)"""
        try:
            # Extract exact size if mentioned
            size_patterns = [
                r'(\d+)\s*(?:sq\s*ft|sqft|square\s*feet|sf)\s*exactly',
                r'exactly\s*(\d+)\s*(?:sq\s*ft|sqft|square\s*feet|sf)',
                r'(\d{4,6})\s*(?:sq\s*ft|sqft|square\s*feet|sf)(?!\s*\+)'  # Large specific sizes without +
            ]
            
            for pattern in size_patterns:
                match = re.search(pattern, query.lower())
                if match:
                    exact_size = int(match.group(1))
                    logger.info(f"Searching for exact size: {exact_size} sq ft")
                    
                    # Search for exact size match
                    cursor.execute('''
                        SELECT unique_id, property_address, floor, suite, size_sf, rent_sf_year, 
                            associate_1, associate_2, broker_email, searchable_text
                        FROM property_listings 
                        WHERE size_sf = ?
                        ORDER BY unique_id
                    ''', (exact_size,))
                    
                    rows = cursor.fetchall()
                    
                    if rows:
                        results = []
                        for row in rows:
                            content = self.format_property_content(row)
                            results.append({
                                'doc_id': f'property_{row[0]}',
                                'content': content,
                                'metadata': {
                                    'unique_id': row[0],
                                    'property_address': row[1],
                                    'size_sf': row[4],
                                    'type': 'property_listing',
                                    'match_type': 'exact_size'
                                },
                                'source_file': 'property_database',
                                'doc_type': 'property_listing',
                                'similarity_score': 1.0
                            })
                        
                        logger.info(f"Found {len(results)} properties with exact size {exact_size} sq ft")
                        return results
            
            return []
            
        except Exception as e:
            logger.error(f"Exact property match search error: {str(e)}")
            return []
    
    async def enhanced_property_search(self, query: str, cursor, top_k: int = 50, user_id: str = None) -> List[Dict[str, Any]]:
        """Enhanced property search with multiple matching strategies"""
        try:
            # Extract search criteria
            criteria = self.extract_search_criteria(query)
            logger.info(f"Extracted criteria: {criteria}")
            
            # Build comprehensive search query
            conditions = []
            params = []
            
            # Size filtering with ranges
            if criteria.get('min_size'):
                conditions.append('size_sf >= ?')
                params.append(criteria['min_size'])
            
            if criteria.get('max_size'):
                conditions.append('size_sf <= ?')
                params.append(criteria['max_size'])
            
            if criteria.get('exact_size'):
                conditions.append('size_sf = ?')
                params.append(criteria['exact_size'])
            
            # Location filtering
            if criteria.get('location'):
                conditions.append('LOWER(property_address) LIKE ?')
                params.append(f"%{criteria['location'].lower()}%")
            
            # Floor filtering
            if criteria.get('floor'):
                conditions.append('LOWER(floor) LIKE ?')
                params.append(f"%{criteria['floor'].lower()}%")
            
            # Keyword search in searchable text
            if criteria.get('keywords'):
                keyword_conditions = []
                for keyword in criteria['keywords']:
                    keyword_conditions.append('LOWER(searchable_text) LIKE ?')
                    params.append(f"%{keyword.lower()}%")
                if keyword_conditions:
                    conditions.append(f"({' OR '.join(keyword_conditions)})")
            
            # Build final query
            where_clause = ''
            if conditions:
                where_clause = 'WHERE ' + ' AND '.join(conditions)
            
            query_sql = f'''
                SELECT unique_id, property_address, floor, suite, size_sf, rent_sf_year, 
                    associate_1, associate_2, broker_email, searchable_text
                FROM property_listings 
                {where_clause}
                ORDER BY 
                    CASE WHEN size_sf IS NOT NULL THEN 0 ELSE 1 END,
                    size_sf DESC
                LIMIT ?
            '''
            params.append(top_k)
            
            cursor.execute(query_sql, params)
            rows = cursor.fetchall()
            
            # Convert to RAG format
            results = []
            for row in rows:
                content = self.format_property_content(row)
                
                # Calculate relevance score
                score = self.calculate_property_relevance(row, criteria, query)
                
                results.append({
                    'doc_id': f'property_{row[0]}',
                    'content': content,
                    'metadata': {
                        'unique_id': row[0],
                        'property_address': row[1],
                        'size_sf': row[4],
                        'floor': row[2],
                        'suite': row[3],
                        'type': 'property_listing'
                    },
                    'source_file': 'property_database',
                    'doc_type': 'property_listing',
                    'similarity_score': score
                })
            
            # Sort by relevance score
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Track user interactions for personalization
            if user_id and results:
                await self.track_user_property_interaction(user_id, query, results[:5], cursor)
            
            logger.info(f"Enhanced property search returned {len(results)} properties")
            return results
            
        except Exception as e:
            logger.error(f"Enhanced property search error: {str(e)}")
            return await self.get_all_properties(cursor, top_k)
    
    def extract_search_criteria(self, query: str) -> Dict[str, Any]:
        """Extract detailed search criteria from query"""
        criteria = {}
        query_lower = query.lower()
        
        # Extract exact size
        exact_size_patterns = [
            r'(\d+)\s*(?:sq\s*ft|sqft|square\s*feet|sf)\s*exactly',
            r'exactly\s*(\d+)\s*(?:sq\s*ft|sqft|square\s*feet|sf)'
        ]
        
        for pattern in exact_size_patterns:
            match = re.search(pattern, query_lower)
            if match:
                criteria['exact_size'] = int(match.group(1))
                break
        
        # Extract size ranges
        if 'exact_size' not in criteria:
            # Min size (e.g., "3000+ sq ft", "at least 2000 sq ft")
            min_size_patterns = [
                r'(\d+)\+\s*(?:sq\s*ft|sqft|square\s*feet|sf)',
                r'(?:at least|minimum|min)\s*(\d+)\s*(?:sq\s*ft|sqft|square\s*feet|sf)',
                r'(\d+)\s*(?:sq\s*ft|sqft|square\s*feet|sf)\s*(?:or more|plus|minimum)'
            ]
            
            for pattern in min_size_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    criteria['min_size'] = int(match.group(1))
                    break
            
            # Max size (e.g., "under 5000 sq ft", "maximum 3000 sq ft")
            max_size_patterns = [
                r'(?:under|below|max|maximum)\s*(\d+)\s*(?:sq\s*ft|sqft|square\s*feet|sf)',
                r'(\d+)\s*(?:sq\s*ft|sqft|square\s*feet|sf)\s*(?:or less|maximum|max)'
            ]
            
            for pattern in max_size_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    criteria['max_size'] = int(match.group(1))
                    break
            
            # Range (e.g., "2000 to 5000 sq ft", "between 1000-3000 sq ft")
            range_patterns = [
                r'(\d+)\s*(?:to|-)\s*(\d+)\s*(?:sq\s*ft|sqft|square\s*feet|sf)',
                r'between\s*(\d+)\s*(?:and|-)\s*(\d+)\s*(?:sq\s*ft|sqft|square\s*feet|sf)'
            ]
            
            for pattern in range_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    criteria['min_size'] = int(match.group(1))
                    criteria['max_size'] = int(match.group(2))
                    break
        
        # Extract location preferences
        location_keywords = ['downtown', 'midtown', 'times square', 'manhattan', 'brooklyn', 
                           'soho', 'tribeca', 'financial district', 'chelsea', 'flatiron']
        
        for location in location_keywords:
            if location in query_lower:
                criteria['location'] = location
                break
        
        # Extract floor preferences
        floor_match = re.search(r'(\d+)(?:st|nd|rd|th)?\s*floor', query_lower)
        if floor_match:
            criteria['floor'] = floor_match.group(1)
        
        # Extract general keywords
        keywords = []
        keyword_patterns = [
            r'\b(parking|garage)\b',
            r'\b(elevator|lift)\b', 
            r'\b(gym|fitness)\b',
            r'\b(restaurant|dining)\b',
            r'\b(conference|meeting)\b',
            r'\b(reception|lobby)\b'
        ]
        
        for pattern in keyword_patterns:
            matches = re.findall(pattern, query_lower)
            keywords.extend(matches)
        
        if keywords:
            criteria['keywords'] = list(set(keywords))
        
        return criteria
    
    def format_property_content(self, row) -> str:
        """Format property data consistently"""
        return (
            f"unique id: {row[0]} | "
            f"property address: {row[1]} | "
            f"floor: {row[2] or 'N/A'} | "
            f"suite: {row[3] or 'N/A'} | "
            f"size (sf): {row[4] or 'N/A'} | "
            f"rent/sf/year: {row[5] or 'Contact for pricing'} | "
            f"associate 1: {row[6] or 'Contact office'} | "
            f"associate 2: {row[7] or 'Contact office'} | "
            f"broker email id: {row[8] or 'info@company.com'} | "
            f"phone: Contact for details"
        )
    
    def calculate_property_relevance(self, row, criteria: Dict[str, Any], original_query: str) -> float:
        """Calculate relevance score for property based on criteria"""
        score = 0.0
        
        # Size matching (highest priority)
        if criteria.get('exact_size') and row[4] == criteria['exact_size']:
            score += 1.0
        elif criteria.get('min_size') and row[4] and row[4] >= criteria['min_size']:
            score += 0.8
        elif criteria.get('max_size') and row[4] and row[4] <= criteria['max_size']:
            score += 0.8
        
        # Location matching
        if criteria.get('location') and row[1]:
            if criteria['location'].lower() in row[1].lower():
                score += 0.6
        
        # Floor matching
        if criteria.get('floor') and row[2]:
            if criteria['floor'] in str(row[2]):
                score += 0.4
        
        # Keyword matching in searchable text
        if criteria.get('keywords') and row[9]:
            keyword_score = 0
            for keyword in criteria['keywords']:
                if keyword.lower() in row[9].lower():
                    keyword_score += 0.2
            score += min(keyword_score, 0.8)
        
        # Fuzzy string matching for address
        if row[1] and len(original_query) > 3:
            fuzzy_score = fuzz.partial_ratio(original_query.lower(), row[1].lower()) / 100
            score += fuzzy_score * 0.3
        
        return min(score, 1.0)
    
    async def track_user_property_interaction(self, user_id: str, query: str, results: List[Dict], cursor):
        """Track user property interactions for personalization"""
        try:
            for result in results:
                interaction_id = str(uuid.uuid4())
                cursor.execute('''
                    INSERT INTO user_property_interactions 
                    (interaction_id, user_id, property_id, interaction_type, details)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    interaction_id,
                    user_id,
                    result['metadata']['unique_id'],
                    'search_result',
                    json.dumps({'query': query, 'score': result['similarity_score']})
                ))
        except Exception as e:
            logger.error(f"Track user interaction error: {str(e)}")
    
    async def get_all_properties(self, cursor, limit: int = 50) -> List[Dict[str, Any]]:
        """Fallback: get all properties with enhanced format"""
        try:
            cursor.execute('''
                SELECT unique_id, property_address, floor, suite, size_sf, rent_sf_year, 
                    associate_1, associate_2, broker_email, searchable_text
                FROM property_listings 
                ORDER BY unique_id
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            results = []
            
            for row in rows:
                content = self.format_property_content(row)
                
                results.append({
                    'doc_id': f'property_{row[0]}',
                    'content': content,
                    'metadata': {
                        'unique_id': row[0],
                        'property_address': row[1],
                        'size_sf': row[4],
                        'type': 'property_listing'
                    },
                    'source_file': 'property_database',
                    'doc_type': 'property_listing',
                    'similarity_score': 0.5
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Get all properties error: {str(e)}")
            return []
    
    async def enhanced_keyword_search(self, query: str, cursor, top_k: int) -> List[Dict[str, Any]]:
        """Enhanced keyword search with better scoring"""
        try:
            keywords = re.findall(r'\w+', query.lower())
            
            if not keywords:
                return []
            
            # Build search query with scoring
            search_conditions = []
            params = []
            
            for keyword in keywords:
                search_conditions.append('LOWER(searchable_content) LIKE ?')
                params.append(f'%{keyword}%')
            
            query_sql = f'''
                SELECT doc_id, content, metadata, source_file, doc_type, searchable_content
                FROM documents 
                WHERE {' OR '.join(search_conditions)}
                LIMIT ?
            '''
            params.append(top_k)
            
            cursor.execute(query_sql, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                # Calculate enhanced keyword match score
                content_lower = (row[1] or '').lower() + ' ' + (row[5] or '').lower()
                score = 0
                for keyword in keywords:
                    if keyword in content_lower:
                        score += 1
                score = score / len(keywords)
                
                results.append({
                    'doc_id': row[0],
                    'content': row[1],
                    'metadata': json.loads(row[2] or '{}'),
                    'source_file': row[3],
                    'doc_type': row[4],
                    'similarity_score': score
                })
            
            # Sort by score
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Enhanced keyword search error: {str(e)}")
            return []
    
    async def semantic_search(self, query_embeddings: List[float], cursor, top_k: int) -> List[Dict[str, Any]]:
        """Enhanced semantic search"""
        try:
            cursor.execute('SELECT doc_id, content, metadata, embeddings, source_file, doc_type FROM documents WHERE embeddings IS NOT NULL')
            rows = cursor.fetchall()
            
            if not rows:
                return []
            
            similarities = []
            for row in rows:
                try:
                    doc_embeddings = pickle.loads(row[3])
                    similarity = cosine_similarity([query_embeddings], [doc_embeddings])[0][0]
                    similarities.append((similarity, row))
                except Exception:
                    continue
            
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            results = []
            for similarity, row in similarities[:top_k]:
                results.append({
                    'doc_id': row[0],
                    'content': row[1],
                    'metadata': json.loads(row[2] or '{}'),
                    'source_file': row[4],
                    'doc_type': row[5],
                    'similarity_score': float(similarity)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search error: {str(e)}")
            return []
    
    async def get_property_addresses_for_dropdown(self) -> List[Dict[str, Any]]:
        """Get all property addresses with floor and suite info for dropdown"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT unique_id, property_address, floor, suite, size_sf
                FROM property_listings 
                ORDER BY property_address, floor, suite
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            properties = []
            for row in rows:
                display_text = row[1]  # property_address
                if row[2]:  # floor
                    display_text += f", Floor {row[2]}"
                if row[3]:  # suite
                    display_text += f", Suite {row[3]}"
                if row[4]:  # size_sf
                    display_text += f" ({row[4]} sq ft)"
                
                properties.append({
                    'id': row[0],
                    'address': row[1],
                    'floor': row[2],
                    'suite': row[3],
                    'size_sf': row[4],
                    'display_text': display_text,
                    'value': f"{row[1]} - Floor {row[2] or 'N/A'}, Suite {row[3] or 'N/A'}"
                })
            
            return properties
            
        except Exception as e:
            logger.error(f"Get property addresses error: {str(e)}")
            return []
    
    async def get_user_property_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user's property preferences based on interaction history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user's recent interactions
            cursor.execute('''
                SELECT property_id, interaction_type, details, timestamp
                FROM user_property_interactions 
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT 10
            ''', (user_id,))
            
            interactions = cursor.fetchall()
            
            if not interactions:
                conn.close()
                return {}
            
            # Analyze preferences
            preferred_sizes = []
            preferred_locations = []
            
            for interaction in interactions:
                try:
                    details = json.loads(interaction[2])
                    if 'query' in details:
                        query = details['query'].lower()
                        
                        # Extract size preferences
                        size_match = re.search(r'(\d+)\s*(?:sq\s*ft|sqft|square\s*feet|sf)', query)
                        if size_match:
                            preferred_sizes.append(int(size_match.group(1)))
                        
                        # Extract location preferences
                        for location in ['downtown', 'midtown', 'times square', 'manhattan']:
                            if location in query:
                                preferred_locations.append(location)
                except:
                    continue
            
            conn.close()
            
            preferences = {}
            if preferred_sizes:
                preferences['preferred_size_range'] = {
                    'min': min(preferred_sizes),
                    'max': max(preferred_sizes),
                    'avg': sum(preferred_sizes) // len(preferred_sizes)
                }
            
            if preferred_locations:
                preferences['preferred_locations'] = list(set(preferred_locations))
            
            return preferences
            
        except Exception as e:
            logger.error(f"Get user preferences error: {str(e)}")
            return {}
    
    def rebuild_tfidf_matrix(self):
        """Rebuild TF-IDF matrix from all documents"""
        if not self.tfidf_vectorizer:
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all documents
            cursor.execute('SELECT content FROM documents WHERE content IS NOT NULL')
            rows = cursor.fetchall()
            
            if rows:
                texts = [row[0] for row in rows]
                self.documents = texts
                
                # Fit TF-IDF vectorizer
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                logger.info(f"TF-IDF matrix rebuilt with {len(texts)} documents")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"TF-IDF matrix rebuild error: {str(e)}")
    
    # Keep existing methods for text, JSON, and PDF loading
    async def load_text_data(self, file_path: str):
        """Load text file data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = self.split_text_into_chunks(content)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for i, chunk in enumerate(chunks):
                embeddings = self.generate_embeddings(chunk)
                doc_id = str(uuid.uuid4())
                
                cursor.execute('''
                    INSERT INTO documents 
                    (doc_id, content, metadata, embeddings, source_file, doc_type, searchable_content)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    doc_id,
                    chunk,
                    json.dumps({"chunk_index": i, "total_chunks": len(chunks)}),
                    pickle.dumps(embeddings) if embeddings else None,
                    file_path,
                    'text_chunk',
                    chunk.lower()
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Loaded {len(chunks)} text chunks from {file_path}")
            
            if self.tfidf_vectorizer:
                self.rebuild_tfidf_matrix()
            
        except Exception as e:
            logger.error(f"Text data loading error: {str(e)}")
            raise
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk.strip()]