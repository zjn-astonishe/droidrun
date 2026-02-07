"""
Trajectory data processing and action merging utilities.

This module provides tools for:
- Loading trajectory data from recorded sessions
- Analyzing UI states and action sequences
- Merging similar actions to create higher-level workflows
- Exporting processed trajectories
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class Action:
    """Represents a single action in the trajectory.
    
    Attributes:
        type: Action type identifier
        action_type: Specific action category (e.g., 'click', 'input', 'swipe')
        description: Human-readable action description
        index: Element index in UI hierarchy
        element_text: Text content of the target element
        text: Input text for text entry actions
        package: Package name for app launch actions
    """
    type: str
    action_type: str
    description: str
    index: Optional[int] = None
    element_text: Optional[str] = None
    text: Optional[str] = None
    package: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Action':
        """Create Action from dictionary"""
        return cls(
            type=data.get('type', ''),
            action_type=data.get('action_type', ''),
            description=data.get('description', ''),
            index=data.get('element_index'),
            element_text=data.get('element_text'),
            text=data.get('text'),
            package=data.get('package')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Action to dictionary"""
        result = {
            'type': self.type,
            'action_type': self.action_type,
            'description': self.description
        }
        if self.index is not None:
            result['element_index'] = self.index
        if self.element_text:
            result['element_text'] = self.element_text
        if self.text:
            result['text'] = self.text
        if self.package:
            result['package'] = self.package
        return result


@dataclass
class MergedAction:
    """Represents a merged action combining multiple atomic actions.
    
    Attributes:
        type: Type of the merged action
        description: High-level description of the merged action
        sub_actions: List of atomic actions that were merged
        metadata: Additional metadata about the merge
    """
    type: str
    description: str
    sub_actions: List[Action]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert MergedAction to dictionary"""
        result = {
            'type': self.type,
            'description': self.description,
            'sub_actions': [action.to_dict() for action in self.sub_actions],
            'action_count': len(self.sub_actions)
        }
        if self.metadata:
            result['metadata'] = self.metadata
        return result


class TrajectoryLoader:
    """Load and parse trajectory data from a directory.
    
    Attributes:
        trajectory_dir: Root directory containing trajectory files
        macro_path: Path to macro.json file
        trajectory_path: Path to trajectory.json file
        ui_states_dir: Directory containing UI state snapshots
    """
    
    def __init__(self, trajectory_dir: Path):
        """Initialize loader with trajectory directory.
        
        Args:
            trajectory_dir: Path to the trajectory directory
            
        Raises:
            FileNotFoundError: If trajectory directory doesn't exist
        """
        self.trajectory_dir = Path(trajectory_dir)
        if not self.trajectory_dir.exists():
            raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")
        
        self.macro_path = self.trajectory_dir / "macro.json"
        self.trajectory_path = self.trajectory_dir / "trajectory.json"
        self.ui_states_dir = self.trajectory_dir / "ui_states"
    
    def load_macro(self) -> Dict[str, Any]:
        """Load macro.json containing action definitions.
        
        Returns:
            Dictionary containing macro data
            
        Raises:
            FileNotFoundError: If macro.json doesn't exist
            json.JSONDecodeError: If macro.json is invalid
        """
        try:
            with open(self.macro_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Macro file not found: {self.macro_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in macro file: {e}")
            raise
    
    def load_trajectory(self) -> List[Dict[str, Any]]:
        """Load trajectory.json containing execution events.
        
        Returns:
            List of trajectory events
            
        Raises:
            FileNotFoundError: If trajectory.json doesn't exist
            json.JSONDecodeError: If trajectory.json is invalid
        """
        try:
            with open(self.trajectory_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Trajectory file not found: {self.trajectory_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in trajectory file: {e}")
            raise
    
    def load_ui_state(self, step: int) -> List[Dict[str, Any]]:
        """Load a specific UI state snapshot.
        
        Args:
            step: Step number (used to construct filename)
            
        Returns:
            List of UI elements, empty if file doesn't exist
        """
        ui_state_path = self.ui_states_dir / f"{step:04d}.json"
        if not ui_state_path.exists():
            return []
        
        try:
            with open(ui_state_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in UI state {step}: {e}")
            return []
    
    def load_all_ui_states(self) -> List[List[Dict[str, Any]]]:
        """Load all UI state snapshots in sequence.
        
        Returns:
            List of UI states, each containing UI elements
        """
        ui_states = []
        step = 0
        
        # Determine max steps by checking directory
        if self.ui_states_dir.exists():
            ui_files = sorted(self.ui_states_dir.glob("*.json"))
            max_steps = len(ui_files)
            logger.info(f"Found {max_steps} UI state files")
        else:
            logger.warning(f"UI states directory not found: {self.ui_states_dir}")
            return ui_states
        
        for step in range(max_steps):
            ui_state = self.load_ui_state(step)
            ui_states.append(ui_state)
        
        return ui_states


class TrajectoryMerger:
    """Merge similar actions in trajectories based on UI state analysis.
    
    Attributes:
        similarity_threshold: Minimum similarity score (0-1) to consider UI states similar
        merge_consecutive_clicks: Whether to merge consecutive clicks on same element
        merge_text_inputs: Whether to merge sequential text inputs
    """
    
    def __init__(
        self, 
        similarity_threshold: float = 0.7,
        merge_consecutive_clicks: bool = True,
        merge_text_inputs: bool = True
    ):
        """Initialize merger with configuration.
        
        Args:
            similarity_threshold: UI similarity threshold (0-1) for merging
            merge_consecutive_clicks: Whether to merge consecutive clicks
            merge_text_inputs: Whether to merge sequential text inputs
            
        Raises:
            ValueError: If similarity_threshold is not between 0 and 1
        """
        if not 0 <= similarity_threshold <= 1:
            raise ValueError(f"similarity_threshold must be between 0 and 1, got {similarity_threshold}")
        
        self.similarity_threshold = similarity_threshold
        self.merge_consecutive_clicks = merge_consecutive_clicks
        self.merge_text_inputs = merge_text_inputs
    
    @lru_cache(maxsize=1024)
    def _cached_ui_similarity(
        self, 
        elements1_tuple: Tuple[Tuple[str, str, str], ...],
        elements2_tuple: Tuple[Tuple[str, str, str], ...]
    ) -> float:
        """Cached calculation of Jaccard similarity between element sets.
        
        Args:
            elements1_tuple: First set of elements as tuple
            elements2_tuple: Second set of elements as tuple
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        elements1 = set(elements1_tuple)
        elements2 = set(elements2_tuple)
        
        intersection = len(elements1 & elements2)
        union = len(elements1 | elements2)
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_ui_similarity(self, ui_state1: List[Dict], ui_state2: List[Dict]) -> float:
        """Calculate similarity between two UI states using Jaccard similarity.
        
        Args:
            ui_state1: First UI state (list of elements)
            ui_state2: Second UI state (list of elements)
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not ui_state1 or not ui_state2:
            return 0.0
        
        # Extract element identifiers and convert to tuples for caching
        elements1 = self._extract_element_identifiers(ui_state1)
        elements2 = self._extract_element_identifiers(ui_state2)
        
        # Use cached similarity calculation
        return self._cached_ui_similarity(
            tuple(sorted(elements1)),
            tuple(sorted(elements2))
        )
    
    def _extract_element_identifiers(self, ui_state: List[Dict]) -> Set[Tuple[str, str, str]]:
        """Extract unique element identifiers from UI state.
        
        Args:
            ui_state: List of UI elements
            
        Returns:
            Set of (resourceId, className, text) tuples
        """
        return {
            (
                e.get('resourceId', ''),
                e.get('className', ''),
                e.get('text', '')[:50]  # Truncate long text for better performance
            )
            for e in ui_state
        }
   
    def calculate_all_similarities(
        self, 
        ui_states: List[List[Dict[str, Any]]]
    ) -> List[List[float]]:
        """Calculate similarity matrix for all UI states.
        
        Args:
            ui_states: List of all UI states in trajectory
            
        Returns:
            2D similarity matrix where [i][j] is similarity between states i and j
        """
        n = len(ui_states)
        if n == 0:
            return []
        
        # Pre-allocate matrix with diagonal set to 1.0
        similarity_matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        
        # Only calculate upper triangle (matrix is symmetric)
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self.calculate_ui_similarity(ui_states[i], ui_states[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        logger.debug(f"Calculated similarity matrix for {n} UI states")
        return similarity_matrix
     
    def merge_actions(
        self, 
        actions: List[Action],
        ui_states: Optional[List[List[Dict[str, Any]]]] = None
    ) -> List[MergedAction]:
        """Merge actions based on patterns and UI similarity.
        
        Args:
            actions: List of actions to merge
            ui_states: Optional list of UI states for context-aware merging
            
        Returns:
            List of merged actions
        """
        if not actions:
            return []
        
        merged_actions = []
        i = 0
        
        while i < len(actions):
            current_action = actions[i]
            
            # Try to merge consecutive text inputs
            if (self.merge_text_inputs and 
                current_action.action_type == 'input_text'):
                merged, next_i = self._merge_text_inputs(actions, i)
                if merged:
                    merged_actions.append(merged)
                    i = next_i
                    continue
            
            # Try to merge consecutive clicks on same element
            if (self.merge_consecutive_clicks and 
                current_action.action_type == 'click'):
                merged, next_i = self._merge_consecutive_clicks(actions, i)
                if merged:
                    merged_actions.append(merged)
                    i = next_i
                    continue
            
            # No merge possible, wrap as single-action merged action
            merged_actions.append(MergedAction(
                type=current_action.action_type,
                description=current_action.description,
                sub_actions=[current_action]
            ))
            i += 1
        
        logger.info(f"Merged {len(actions)} actions into {len(merged_actions)} merged actions")
        return merged_actions
    
    def _merge_text_inputs(
        self, 
        actions: List[Action], 
        start_idx: int
    ) -> Tuple[Optional[MergedAction], int]:
        """Merge consecutive text input actions.
        
        Args:
            actions: List of all actions
            start_idx: Starting index
            
        Returns:
            Tuple of (merged action or None, next index)
        """
        sub_actions = [actions[start_idx]]
        i = start_idx + 1
        
        # Collect consecutive text inputs
        while i < len(actions) and actions[i].action_type == 'input_text':
            sub_actions.append(actions[i])
            i += 1
        
        # Only merge if we have multiple text inputs
        if len(sub_actions) > 1:
            combined_text = ' '.join(a.text or '' for a in sub_actions)
            merged = MergedAction(
                type='input_text_sequence',
                description=f"Enter text: {combined_text[:50]}{'...' if len(combined_text) > 50 else ''}",
                sub_actions=sub_actions,
                metadata={'combined_text': combined_text}
            )
            return merged, i
        
        return None, start_idx + 1
    
    def _merge_consecutive_clicks(
        self, 
        actions: List[Action], 
        start_idx: int
    ) -> Tuple[Optional[MergedAction], int]:
        """Merge consecutive clicks on the same element.
        
        Args:
            actions: List of all actions
            start_idx: Starting index
            
        Returns:
            Tuple of (merged action or None, next index)
        """
        first_action = actions[start_idx]
        sub_actions = [first_action]
        i = start_idx + 1
        
        # Collect consecutive clicks on same element
        while i < len(actions) and actions[i].action_type == 'click':
            if (actions[i].index == first_action.index and 
                actions[i].element_text == first_action.element_text):
                sub_actions.append(actions[i])
                i += 1
            else:
                break
        
        # Only merge if we have multiple clicks
        if len(sub_actions) > 1:
            merged = MergedAction(
                type='multi_click',
                description=f"Click {len(sub_actions)} times on: {first_action.element_text or 'element'}",
                sub_actions=sub_actions,
                metadata={'click_count': len(sub_actions)}
            )
            return merged, i
        
        return None, start_idx + 1
    
    def process_trajectory(
        self, 
        trajectory_dir: Path
    ) -> Dict[str, Any]:
        """Process a trajectory and merge similar actions.
        
        Args:
            trajectory_dir: Path to trajectory directory
        
        Returns:
            Dictionary with merged trajectory data
            
        Raises:
            FileNotFoundError: If trajectory directory or files don't exist
        """
        logger.info(f"Processing trajectory: {trajectory_dir}")
        
        try:
            loader = TrajectoryLoader(trajectory_dir)
            macro = loader.load_macro()
            ui_states = loader.load_all_ui_states()
            
            # Convert macro actions to Action objects
            actions = [Action.from_dict(action) for action in macro.get('actions', [])]
            logger.info(f"Loaded {len(actions)} actions")
            
            # Merge actions
            merged_actions = self.merge_actions(actions, ui_states)
            
            # Calculate statistics
            reduction_pct = (
                ((len(actions) - len(merged_actions)) / len(actions) * 100)
                if actions else 0
            )
            
            # Create output structure
            result = {
                'version': '1.0',
                'description': macro.get('description', ''),
                'timestamp': macro.get('timestamp', ''),
                'original_action_count': len(actions),
                'merged_action_count': len(merged_actions),
                'reduction_percentage': round(reduction_pct, 2),
                'ui_state_count': len(ui_states),
                'merged_actions': [action.to_dict() for action in merged_actions],
                'statistics': self._calculate_merge_statistics(merged_actions)
            }
            
            logger.info(
                f"Merged {len(actions)} actions into {len(merged_actions)} "
                f"({reduction_pct:.1f}% reduction)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing trajectory: {e}")
            raise
    
    def _calculate_merge_statistics(
        self, 
        merged_actions: List[MergedAction]
    ) -> Dict[str, Any]:
        """Calculate statistics about merged actions.
        
        Args:
            merged_actions: List of merged actions
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_merged': len(merged_actions),
            'action_types': {},
            'avg_actions_per_merge': 0.0,
            'max_actions_in_merge': 0
        }
        
        total_sub_actions = 0
        
        for merged in merged_actions:
            action_type = merged.type
            stats['action_types'][action_type] = stats['action_types'].get(action_type, 0) + 1
            
            sub_count = len(merged.sub_actions)
            total_sub_actions += sub_count
            stats['max_actions_in_merge'] = max(stats['max_actions_in_merge'], sub_count)
        
        if merged_actions:
            stats['avg_actions_per_merge'] = round(total_sub_actions / len(merged_actions), 2)
        
        return stats


def merge_trajectory(
    trajectory_dir: str, 
    output_path: Optional[str] = None,
    similarity_threshold: float = 0.7,
    merge_consecutive_clicks: bool = True,
    merge_text_inputs: bool = True
) -> Dict[str, Any]:
    """Main function to merge actions in a trajectory.
    
    Args:
        trajectory_dir: Path to trajectory directory
        output_path: Optional path to save merged trajectory
        similarity_threshold: UI similarity threshold for merging (0-1)
        merge_consecutive_clicks: Whether to merge consecutive clicks
        merge_text_inputs: Whether to merge sequential text inputs
    
    Returns:
        Merged trajectory data
        
    Raises:
        FileNotFoundError: If trajectory directory doesn't exist
    """
    merger = TrajectoryMerger(
        similarity_threshold=similarity_threshold,
        merge_consecutive_clicks=merge_consecutive_clicks,
        merge_text_inputs=merge_text_inputs
    )
    
    result = merger.process_trajectory(Path(trajectory_dir))
    
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved merged trajectory to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save merged trajectory: {e}")
            raise
    
    return result


def analyze_trajectory(trajectory_dir: str) -> Dict[str, Any]:
    """Analyze a trajectory and provide detailed statistics.
    
    Args:
        trajectory_dir: Path to trajectory directory
    
    Returns:
        Dictionary with analysis results including action counts, types, and UI states
        
    Raises:
        FileNotFoundError: If trajectory directory doesn't exist
    """
    try:
        loader = TrajectoryLoader(Path(trajectory_dir))
        macro = loader.load_macro()
        trajectory = loader.load_trajectory()
        ui_states = loader.load_all_ui_states()
        
        actions = macro.get('actions', [])
        
        # Count action types
        action_types = {}
        for action in actions:
            action_type = action.get('action_type', 'unknown')
            action_types[action_type] = action_types.get(action_type, 0) + 1
        
        # Analyze UI state transitions
        ui_transition_count = len(ui_states) - 1 if len(ui_states) > 1 else 0
        
        return {
            'total_actions': len(actions),
            'action_types': action_types,
            'ui_state_count': len(ui_states),
            'ui_transitions': ui_transition_count,
            'trajectory_events': len(trajectory),
            'description': macro.get('description', ''),
            'has_macro': bool(actions),
            'has_ui_states': bool(ui_states),
            'trajectory_path': str(Path(trajectory_dir).absolute())
        }
        
    except Exception as e:
        logger.error(f"Error analyzing trajectory: {e}")
        raise

def main():
    """Main entry point for trajectory analysis and merging."""
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example: Analyze a trajectory
    try:
        trajectory_path = "output/trajectories/20260205_193329_e2a18059"
        
        print("=" * 60)
        print("Trajectory Analysis")
        print("=" * 60)
        
        result = analyze_trajectory(trajectory_path)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        print("\n" + "=" * 60)
        print("Merging Trajectory")
        print("=" * 60)
        
        merged = merge_trajectory(
            trajectory_path,
            output_path="output/merged_trajectory.json"
        )
        
        print(f"\nOriginal actions: {merged['original_action_count']}")
        print(f"Merged actions: {merged['merged_action_count']}")
        print(f"Reduction: {merged['reduction_percentage']}%")
        print(f"\nMerged trajectory saved to: output/merged_trajectory.json")
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
