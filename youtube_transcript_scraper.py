import pandas as pd
import numpy as np
import time
import json
import logging
import os
import concurrent.futures
import random
import traceback
import urllib.request
import sys
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
from youtube_transcript_api._errors import RequestBlocked, YouTubeRequestFailed, VideoUnplayable
from datetime import datetime
from json.decoder import JSONDecodeError

def debug_transcript_request(video_id):
    """
    Debug function to test a specific video ID and show the YouTube API request.
    
    Args:
        video_id: YouTube video ID to test
    """
    print(f"\nDebugging transcript request for video ID: {video_id}")
    print("=" * 80)
    
    try:
        # First try to get the video info to see what YouTube returns
        video_info_url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"Video URL: {video_info_url}")
        
        # Inspect YouTube Transcript API source code
        print("\nExamining YouTube Transcript API's actual workflow:")
        print("1. The API first loads the YouTube watch page HTML")
        print("   URL: https://www.youtube.com/watch?v=" + video_id)
        print("2. It extracts 'ytInitialPlayerResponse' JSON from the HTML")
        print("3. From this JSON, it extracts 'captions' data containing URLs")
        print("4. It makes requests to those caption/transcript URLs")
        
        # Use a modified approach with direct urllib request to simulate what the API does
        print("\nAttempting direct access to examine response structure...")
        import urllib.request
        import re
        import json
        
        # Setup headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            # Create a request
            req = urllib.request.Request(video_info_url, headers=headers)
            with urllib.request.urlopen(req) as response:
                html = response.read().decode('utf-8')
                
            print(f"Successfully retrieved HTML page ({len(html)} bytes)")
            
            # Look for player response JSON
            player_json_match = re.search(r'ytInitialPlayerResponse\s*=\s*({.+?});', html)
            if player_json_match:
                json_str = player_json_match.group(1)
                print(f"Found ytInitialPlayerResponse ({len(json_str)} bytes)")
                
                # Try to parse JSON to see if it's valid
                try:
                    player_data = json.loads(json_str)
                    print("Successfully parsed player JSON")
                    
                    # Check for captions data
                    if 'captions' in player_data:
                        captions_data = player_data['captions']
                        print("Found captions data in player response")
                        
                        if 'playerCaptionsTracklistRenderer' in captions_data:
                            tracks = captions_data['playerCaptionsTracklistRenderer'].get('captionTracks', [])
                            print(f"Found {len(tracks)} caption tracks")
                            
                            for i, track in enumerate(tracks):
                                lang = track.get('languageCode', 'unknown')
                                name = track.get('name', {}).get('simpleText', 'Unknown')
                                base_url = track.get('baseUrl', 'No URL')
                                print(f"Track {i+1}: {name} ({lang})")
                                print(f"Base URL: {base_url[:100]}..." if len(base_url) > 100 else f"Base URL: {base_url}")
                                print("---")
                    else:
                        print("No captions data found in player response")
                except json.JSONDecodeError as e:
                    print(f"Error parsing player JSON: {str(e)}")
                    print(f"JSON starts with: {json_str[:100]}...")
            else:
                print("Could not find ytInitialPlayerResponse in the HTML")
        except Exception as e:
            print(f"Error making direct request: {str(e)}")
        
        # Get transcript list through the API
        print("\nAttempting to get transcript list...")
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        print("Available transcript languages:")
        for transcript in transcript_list:
            print(f"- {transcript.language} ({transcript.language_code}) [{'auto-generated' if transcript.is_generated else 'manual'}]")
        
        # Try to get English or auto-generated transcript
        print("\nAttempting to get transcript content...")
        try:
            transcript = transcript_list.find_transcript(['en'])
            print(f"Found transcript in: {transcript.language_code}")
        except:
            print("No English transcript found, trying auto-generated...")
            transcript = transcript_list.find_generated_transcript(['en'])
            print(f"Found auto-generated transcript in: {transcript.language_code}")
        
        # Fetch the transcript
        transcript_data = transcript.fetch()
        print(f"\nSuccessfully retrieved transcript with {len(transcript_data)} segments")
        print("First segment:", transcript_data[0])
        
    except Exception as e:
        print(f"\nError: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nStacktrace:")
        traceback.print_exc()
    
    print("\nSummary of YouTube Transcript API workflow:")
    print("1. Loads watch page HTML: https://www.youtube.com/watch?v=" + video_id)
    print("2. Extracts player config JSON containing caption information")
    print("3. From caption data, extracts the baseUrl for each language track")
    print("4. Makes requests to these baseUrls to get the actual transcript data")
    print("5. The JSONDecodeError happens when parsing the response from one of these URLs")
    
    print("\nTo see the exact URLs being requested:")
    print("Use a network monitoring tool while accessing the video")
    
    print("=" * 80)

# Example usage
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "debug":
    if len(sys.argv) > 2:
        video_id = sys.argv[2]
    else:
        video_id = "fGXIEV0_SMM"  # The video ID that had the JSONDecodeError
    debug_transcript_request(video_id)
    sys.exit(0)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transcript_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Top languages by usage to try when scraping
LANGUAGES = [
    'en','es','fr', 'de', 'pt', 'it',  # European languages
    'ru', 'ja', 'ko', 'zh-CN', 'zh-TW',  # Asian languages
    'ar', 'hi', 'vi', 'id', 'th',  # Other widely used languages
    'tr', 'pl', 'nl', 'sv', 'ro',  # More European languages
    'cs', 'hu', 'el', 'da', 'fi',  # More European languages
    'no', 'sk', 'uk', 'hr', 'ca',  # More European languages
    'he', 'bg', 'lt', 'sl', 'et',  # More European languages
    'ms', 'tl', 'fa', 'ur', 'bn'   # Various other languages
]

class TranscriptScraper:
    def __init__(self, 
                output_dir='transcript_data', 
                max_workers=4, 
                rate_limit=12000,  # Requests per minute
                cooldown_time=5,  # 5 seconds in seconds
                unlimited_mode=True,
                checkpoint_interval=500,
                error_threshold=0.7,  # 70% error rate triggers cooldown
                error_window=50):     # Calculate error rate over this many recent requests
        """
        Initialize the TranscriptScraper.
        
        Args:
            output_dir: Directory to save transcript data
            max_workers: Maximum number of parallel workers
            rate_limit: Maximum requests per minute to avoid IP ban
            cooldown_time: Time to wait in seconds when rate limited
            unlimited_mode: If True, will retry after cooldown when rate limited
            checkpoint_interval: Number of videos to process before saving a checkpoint
            error_threshold: Error rate (0.0-1.0) that triggers a cooldown
            error_window: Number of recent requests to calculate error rate from
        """
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self.cooldown_time = cooldown_time
        self.unlimited_mode = unlimited_mode
        self.checkpoint_interval = checkpoint_interval
        self.error_threshold = error_threshold
        self.error_window = error_window
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'transcripts'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'errors'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        
        # Rate limiting tracking
        self.request_timestamps = []
        
        # Statistics for tracking progress
        self.stats = {
            'total': 0,
            'success': 0,
            'error': 0,
            'no_transcript': 0,
            'video_unavailable': 0,
            'transcript_disabled': 0,
            'too_many_requests': 0,
            'json_decode_errors': 0,
            'other_errors': 0
        }
        
        # Error tracking
        self.errors = {}
        
        # Recent results tracking for error rate calculation
        self.recent_results = []
        
        # Initialize YouTubeTranscriptApi
        self.ytt_api = YouTubeTranscriptApi()
        self.request_count = 0  # Add request counter
    
    def _rate_limit_check(self):
        """
        Check if the current request would exceed the rate limit.
        Remove timestamps older than 1 minute and check if the number of recent
        requests exceeds the rate limit.
        
        Returns:
            bool: True if we're under the rate limit, False otherwise
        """
        current_time = time.time()
        # Remove timestamps older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps if current_time - ts < 60]
        
        if len(self.request_timestamps) >= self.rate_limit:
            return False
        return True
    
    def _wait_for_rate_limit(self):
        """
        Wait until we're under the rate limit again.
        """
        while not self._rate_limit_check():
            time.sleep(1)  # Sleep for 1 second and check again
        
        # Add current timestamp to the list
        self.request_timestamps.append(time.time())
    
    def _ensure_json_serializable(self, transcript_data):
        """
        Ensure transcript data is JSON serializable by converting to standard Python types.
        
        Args:
            transcript_data: Raw transcript data from YouTube API
            
        Returns:
            list: JSON serializable transcript data
        """
        serializable_data = []
        
        # Handle FetchedTranscript objects
        if hasattr(transcript_data, 'snippets'):
            for snippet in transcript_data.snippets:
                serializable_item = {
                    'text': str(snippet.text),
                    'start': float(snippet.start),
                    'duration': float(snippet.duration)
                }
                serializable_data.append(serializable_item)
            return serializable_data
        
        # Handle list of dictionaries (original format)
        for item in transcript_data:
            # Convert each transcript segment to a plain dictionary with primitive types
            serializable_item = {
                'text': str(item.get('text', '')),
                'start': float(item.get('start', 0.0)),
                'duration': float(item.get('duration', 0.0))
            }
            serializable_data.append(serializable_item)
        return serializable_data
    
    def _recreate_api(self):
        """
        Recreate the YouTubeTranscriptApi instance to reset the session.
        """
        self.ytt_api = YouTubeTranscriptApi()
        self.request_count = 0
        logger.info("Recreated YouTubeTranscriptApi instance to reset session")
    
    def get_transcript(self, video_id, get_generated=True):
        """
        Get auto-generated transcript for a video with proper error handling and rate limiting.
        Only retrieves auto-generated transcripts.
        
        Args:
            video_id: YouTube video ID
            get_generated: If True, get auto-generated transcript, otherwise get manually created transcript
        Returns:
            dict: Transcript data or error information
        """
        # Check rate limit
        self._wait_for_rate_limit()
        
        # Increment request counter and recreate API if needed
        self.request_count += 1
        if self.request_count >= 1000:
            self._recreate_api()
        
        result = {
            'video_id': video_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error_type': None,
            'error_message': None,
            'transcripts': []
        }
        
        try:
            if get_generated:
                # Try to get transcript directly
                transcript = self.ytt_api.list(video_id).find_generated_transcript(LANGUAGES)
            else:
                # Try to get transcript directly
                transcript = self.ytt_api.list(video_id).find_manually_created_transcript(LANGUAGES)
            language = transcript.language
            language_code = transcript.language_code
            transcript = transcript.fetch(video_id)
            if transcript:
                # Convert transcript data to JSON serializable format
                serializable_data = self._ensure_json_serializable(transcript)
                
                # Since we're getting directly, we don't know if it's auto-generated
                # We'll assume it's in the default language (usually the video's language)
                result['transcripts'].append({
                    'language': language,
                    'language_code': language_code,
                    'is_generated': get_generated,
                    'data': serializable_data
                })
                
                result['status'] = 'success'
            else:
                result['status'] = 'error'
                result['error_type'] = 'no_transcript_found'
                result['error_message'] = 'No transcript could be fetched'
                
        except RequestBlocked as e:
            result['error_type'] = 'request_blocked'
            result['error_message'] = str(e)
            
        except VideoUnavailable as e:
            result['error_type'] = 'video_unavailable'
            result['error_message'] = str(e)
            
        except TranscriptsDisabled as e:
            if get_generated:
                return self.get_transcript(video_id, get_generated=False)
            else:
                result['error_type'] = 'transcripts_disabled'
                result['error_message'] = str(e)
            
        except NoTranscriptFound as e:
            if get_generated:
                return self.get_transcript(video_id, get_generated=False)
            else:
                result['error_type'] = 'no_transcript_found'
                result['error_message'] = str(e)
        except YouTubeRequestFailed as e:
            result['error_type'] = 'youtube_request_failed'
            result['error_message'] = str(e)

        except VideoUnplayable as e:
            result['error_type'] = 'video_unplayable'
            result['error_message'] = str(e)
        
        except JSONDecodeError as e:
            result['error_type'] = 'json_decode_error'
            result['error_message'] = f"JSONDecodeError: {str(e)}"
            # Just record the error and continue, no retry

        except Exception as e:
            result['error_type'] = 'other'
            result['error_message'] = f"{type(e).__name__}: {str(e)}"
            
        return result
    
    def _update_stats(self, result):
        """
        Update statistics based on the result of transcript fetching.
        
        Args:
            result: Result dictionary from get_transcript
        """
        self.stats['total'] += 1
        
        # Add to recent results for error rate calculation
        is_success = result['status'] == 'success'
        self.recent_results.append(is_success)
        # Keep only the most recent window of results
        if len(self.recent_results) > self.error_window:
            self.recent_results.pop(0)
        
        if is_success:
            self.stats['success'] += 1
        else:
            self.stats['error'] += 1
            
            if result['error_type'] == 'no_transcript_found' or result['error_type'] == 'no_auto_transcript_found':
                self.stats['no_transcript'] += 1
            elif result['error_type'] == 'video_unavailable':
                self.stats['video_unavailable'] += 1
            elif result['error_type'] == 'transcripts_disabled':
                self.stats['transcript_disabled'] += 1
            elif result['error_type'] == 'too_many_requests' or result['error_type'] == 'request_blocked':
                self.stats['too_many_requests'] += 1
            elif result['error_type'] == 'json_decode_error':
                self.stats['json_decode_errors'] += 1
            else:
                self.stats['other_errors'] += 1
            
            # Track specific errors
            if result['error_type'] not in self.errors:
                self.errors[result['error_type']] = []
            self.errors[result['error_type']].append(result['video_id'])
    
    def _get_current_error_rate(self):
        """
        Calculate the current error rate based on recent results.
        
        Returns:
            float: Error rate between 0.0 and 1.0
        """
        if not self.recent_results:
            return 0.0
        
        # Calculate the proportion of errors in recent results
        error_count = sum(1 for success in self.recent_results if not success)
        return error_count / len(self.recent_results)
    
    def _save_transcript(self, result):
        """
        Save transcript data to disk.
        
        Args:
            result: Result dictionary from get_transcript
        """
        video_id = result['video_id']
        
        try:
            if result['status'] == 'success':
                # Save successful transcript
                with open(os.path.join(self.output_dir, 'transcripts', f"{video_id}.json"), 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
            else:
                # Save error information
                with open(os.path.join(self.output_dir, 'errors', f"{video_id}.json"), 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Error saving transcript for video {video_id}: {str(e)}")
    
    def _save_checkpoint(self, processed_ids):
        """
        Save a checkpoint of processed video IDs.
        
        Args:
            processed_ids: Set of processed video IDs
        """
        checkpoint_file = os.path.join(self.output_dir, 'checkpoints', f"checkpoint_{int(time.time())}.json")
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats,
            'processed_ids': list(processed_ids)
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Checkpoint saved: {checkpoint_file}")
        
        # Also save current stats
        with open(os.path.join(self.output_dir, 'stats.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats,
                'errors': {k: len(v) for k, v in self.errors.items()}
            }, f, ensure_ascii=False, indent=4)
    
    def load_checkpoint(self, checkpoint_file):
        """
        Load a checkpoint to resume processing.
        
        Args:
            checkpoint_file: Path to the checkpoint file
            
        Returns:
            set: Set of processed video IDs
        """
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # Update stats from checkpoint
            self.stats = checkpoint_data['stats']
            
            logger.info(f"Loaded checkpoint with {len(checkpoint_data['processed_ids'])} processed videos")
            return set(checkpoint_data['processed_ids'])
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return set()
    
    def find_latest_checkpoint(self):
        """
        Find the latest checkpoint file.
        
        Returns:
            str or None: Path to the latest checkpoint file, or None if no checkpoints exist
        """
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')]
        
        if not checkpoint_files:
            return None
        
        # Get the most recent checkpoint file based on modification time
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        return latest_checkpoint
    
    def process_video_batch(self, video_ids, resume=True):
        """
        Process a batch of video IDs, with option to resume from checkpoint.
        
        Args:
            video_ids: List of video IDs to process
            resume: If True, try to resume from the latest checkpoint
            
        Returns:
            set: Set of processed video IDs
        """
        processed_ids = set()
        
        # Try to resume from checkpoint if requested
        if resume:
            latest_checkpoint = self.find_latest_checkpoint()
            if latest_checkpoint:
                processed_ids = self.load_checkpoint(latest_checkpoint)
        
        # Filter out already processed IDs
        video_ids_to_process = [vid for vid in video_ids if vid not in processed_ids]
        
        logger.info(f"Starting processing of {len(video_ids_to_process)} videos")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            # Submit all tasks
            for video_id in video_ids_to_process:
                futures.append(executor.submit(self.get_transcript, video_id))
            
            # Process results as they complete
            pbar = tqdm(total=len(futures), desc="Processing videos")
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result()
                    self._update_stats(result)
                    self._save_transcript(result)
                    processed_ids.add(result['video_id'])
                    
                    # Update progress bar with success/error stats
                    pbar.set_description(f"Processing videos | Success: {self.stats['success']} | Errors: {self.stats['error']}")
                    pbar.update(1)
                    
                    # Save checkpoint at regular intervals
                    if i > 0 and i % self.checkpoint_interval == 0:
                        self._save_checkpoint(processed_ids)
                        
                except Exception as e:
                    logger.error(f"Error processing a video: {str(e)}")
                    pbar.update(1)
            
            pbar.close()
        
        # Save final checkpoint
        self._save_checkpoint(processed_ids)

        return processed_ids
    
    def display_stats(self):
        """
        Display current statistics.
        """
        total = self.stats['total']
        if total == 0:
            logger.info("No videos processed yet")
            return
        
        success_rate = (self.stats['success'] / total) * 100
        
        logger.info(f"--- Statistics ---")
        logger.info(f"Total videos processed: {total}")
        logger.info(f"Successful: {self.stats['success']} ({success_rate:.2f}%)")
        logger.info(f"Failed: {self.stats['error']} ({100 - success_rate:.2f}%)")
        logger.info(f"- No transcript: {self.stats['no_transcript']}")
        logger.info(f"- Video unavailable: {self.stats['video_unavailable']}")
        logger.info(f"- Transcript disabled: {self.stats['transcript_disabled']}")
        logger.info(f"- Rate limited: {self.stats['too_many_requests']}")
        logger.info(f"- JSON decode errors: {self.stats['json_decode_errors']}")
        logger.info(f"- Other errors: {self.stats['other_errors']}")
    
    def get_unique_video_ids(self, sponsor_df):
        """
        Extract unique video IDs from the SponsorBlock dataset.
        
        Args:
            sponsor_df: DataFrame containing SponsorBlock data
            
        Returns:
            list: List of unique video IDs
        """
        return sponsor_df['videoID'].unique().tolist()
    
    def convert_to_dataframe(self):
        """
        Convert all fetched transcripts into a pandas DataFrame for analysis.
        
        Returns:
            DataFrame: Transcript data in pandas DataFrame format
        """
        transcript_dir = os.path.join(self.output_dir, 'transcripts')
        transcript_files = [os.path.join(transcript_dir, f) for f in os.listdir(transcript_dir) if f.endswith('.json')]
        
        transcript_data = []
        
        for file_path in transcript_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                video_id = data['video_id']
                
                for transcript in data['transcripts']:
                    language = transcript['language']
                    language_code = transcript['language_code']
                    is_generated = transcript['is_generated']
                    for item in transcript['data']:
                        transcript_data.append({
                            'video_id': video_id,
                            'language': language,
                            'language_code': language_code,
                            'is_generated': is_generated,
                            'text': item['text'],
                            'start': item['start'],
                            'duration': item['duration'],
                            'end': item['start'] + item['duration']
                        })
            except Exception as e:
                logger.error(f"Error processing transcript file {file_path}: {str(e)}")
                continue
        
        return pd.DataFrame(transcript_data)
    
    def merge_with_sponsor_data(self, transcript_df, sponsor_df):
        """
        Merge transcript data with SponsorBlock data.
        
        Args:
            transcript_df: DataFrame containing transcript data
            sponsor_df: DataFrame containing SponsorBlock data
            
        Returns:
            DataFrame: Merged data
        """
        # Create a copy of sponsor_df with renamed columns for clarity
        sponsor_clean = sponsor_df.copy()
        sponsor_clean = sponsor_clean.rename(columns={
            'startTime': 'sponsor_start',
            'endTime': 'sponsor_end',
            'category': 'sponsor_category',
            'actionType': 'sponsor_action'
        })
        
        # Select relevant columns
        sponsor_clean = sponsor_clean[['videoID', 'sponsor_start', 'sponsor_end', 'sponsor_category', 'sponsor_action']]
        
        # Merge the dataframes
        merged_df = pd.merge(
            transcript_df,
            sponsor_clean,
            left_on='video_id',
            right_on='videoID',
            how='left'
        )
        
        # Create a column to indicate if the text segment overlaps with a sponsor segment
        merged_df['is_sponsor'] = (
            (merged_df['start'] <= merged_df['sponsor_end']) & 
            (merged_df['end'] >= merged_df['sponsor_start'])
        )
        
        return merged_df 