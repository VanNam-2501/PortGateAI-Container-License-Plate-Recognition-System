import re
import time
import datetime
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple
from app.schemas import RecognitionStageResult, TransactionResult, RecognizedObject
from app.utils.validators.container_validator import normalize_container_for_voting

class AggregationStage:
    """Tầng 5: Aggregation & Association - Gom nhóm kết quả từ nhiều frame hoặc camera."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("aggregation", {})
        self.enabled = self.config.get("enabled", True)
        self.strategy = self.config.get("voting_strategy", "majority")
        self.min_frames = self.config.get("min_frames", 1)

    def run(self, stage_results: List[RecognitionStageResult]) -> TransactionResult:
        """Aggregates multiple recognition frame results into a single transaction result.
        
        Args:
            stage_results: A list of RecognitionStageResult objects collected over multiple frames.
            
        Returns:
            TransactionResult containing aggregated container codes and license plates.
        """
        start_time = time.perf_counter()
        
        if not stage_results:
            return TransactionResult(
                status="failed",
                lane_id="UNKNOWN",
                timestamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))),
                total_latency_ms=0.0,
                plates=[],
                containers=[],
                stages_latency={}
            )
            
        # Get metadata from the first result
        first_ingestion = stage_results[0].transformation_result.detection_result.ingestion
        lane_id = first_ingestion.lane_id
        timestamp = first_ingestion.timestamp
        
        # Calculate stages latency sums
        latencies = defaultdict(float)
        for res in stage_results:
            latencies["ingestion"] += 0.0  # Quality check happens inside ingestion
            latencies["detection"] += res.transformation_result.detection_result.latency_ms
            latencies["transformation"] += res.transformation_result.latency_ms
            latencies["recognition"] += res.latency_ms
            
        # Classify objects across all frames
        plates_list: List[RecognizedObject] = []
        containers_list: List[RecognizedObject] = []
        iso_types_list: List[RecognizedObject] = []
        
        for res in stage_results:
            for obj in res.recognized_objects:
                cls = obj.detection.class_name
                if cls == "plate":
                    plates_list.append(obj)
                elif cls == "container_code":
                    containers_list.append(obj)
                elif cls == "iso_type":
                    iso_types_list.append(obj)
                    
        # Aggregate logic
        aggregated_plates = self._aggregate_class(plates_list, class_type="plate")
        aggregated_containers = self._aggregate_class(containers_list, class_type="container_code")
        aggregated_iso_types = self._aggregate_class(iso_types_list, class_type="iso_type")
        
        # Associate ISO types to containers if possible
        # For simplicity, if we have 1 container and 1 ISO type, we pair them.
        # In a advanced system, we would pair based on OBB distance/alignment.
        container_payloads = []
        for i, container in enumerate(aggregated_containers):
            # Try to associate with ISO type
            iso_type_val = ""
            iso_conf = 0.0
            if i < len(aggregated_iso_types):
                iso_type_val = aggregated_iso_types[i]["text"]
                iso_conf = aggregated_iso_types[i]["confidence"]
                
            container_payloads.append({
                "container_code": container["text"],
                "confidence": container["confidence"],
                "is_valid": container["is_valid"],
                "validation_message": container["validation_message"],
                "iso_type": iso_type_val,
                "iso_confidence": iso_conf
            })
            
        # Determine status
        status = "success"
        if not aggregated_plates and not container_payloads:
            status = "failed"
        elif not aggregated_plates or not container_payloads:
            status = "partial"
            
        aggregation_latency = (time.perf_counter() - start_time) * 1000.0
        latencies["aggregation"] = aggregation_latency
        total_latency = sum(latencies.values())
        
        return TransactionResult(
            status=status,
            lane_id=lane_id,
            timestamp=timestamp,
            total_latency_ms=total_latency,
            plates=aggregated_plates,
            containers=container_payloads,
            stages_latency=dict(latencies)
        )

    def _aggregate_class(self, objects: List[RecognizedObject], class_type: str = "") -> List[Dict[str, Any]]:
        """Helper to aggregate a specific class list (plates, containers, etc.)."""
        if not objects:
            return []
            
        # Group by normalized text value
        text_stats = defaultdict(list)
        for obj in objects:
            if obj.recognition.text:
                # Normalize container codes and ISO types for better grouping
                key = obj.recognition.text
                if class_type == "container_code":
                    key = normalize_container_for_voting(obj.recognition.text)
                elif class_type == "iso_type":
                    cleaned = re.sub(r'[^A-Z0-9]', '', obj.recognition.text.upper())
                    if len(cleaned) >= 4 and cleaned[0].isdigit() and cleaned[1].isdigit():
                        key = cleaned[:4]  # Normalize ISO type to 4 chars
                    else:
                        key = cleaned
                text_stats[key].append(obj)
                
        if not text_stats:
            return []
            
        aggregated = []
        
        if self.strategy == "majority":
            # Sort by frequency (majority voting)
            # We want to favor valid ones if there is a tie or near-tie
            sorted_texts = sorted(
                text_stats.keys(),
                key=lambda t: (
                    sum(1 for o in text_stats[t] if o.recognition.is_valid), # Number of valid frames
                    len(text_stats[t]),                                      # Total frequency
                    max(o.recognition.confidence for o in text_stats[t])     # Max confidence fallback
                ),
                reverse=True
            )
            
            # Select top text (we assume 1 active object of this class per transaction in simple ports,
            # or handle multiple if they don't overlap)
            # Let's return the top-1 result for now
            top_text = sorted_texts[0]
            top_objs = text_stats[top_text]
            
            # Calculate average confidence
            avg_conf = sum(o.recognition.confidence for o in top_objs) / len(top_objs)
            # Find the best valid check flag and validation message
            is_valid = any(o.recognition.is_valid for o in top_objs)
            val_msg = next((o.recognition.validation_message for o in top_objs if o.recognition.is_valid), top_objs[0].recognition.validation_message)
            
            aggregated.append({
                "text": top_text,
                "confidence": float(avg_conf),
                "is_valid": is_valid,
                "validation_message": val_msg,
                "frames_count": len(top_objs)
            })
            
        elif self.strategy == "best_confidence":
            # Sort by absolute max confidence
            flat_list = []
            for text, objs in text_stats.items():
                best_obj = max(objs, key=lambda o: o.recognition.confidence)
                flat_list.append(best_obj)
                
            best_overall = max(flat_list, key=lambda o: o.recognition.confidence)
            aggregated.append({
                "text": best_overall.recognition.text,
                "confidence": float(best_overall.recognition.confidence),
                "is_valid": best_overall.recognition.is_valid,
                "validation_message": best_overall.recognition.validation_message,
                "frames_count": len(text_stats[best_overall.recognition.text])
            })
            
        return aggregated
