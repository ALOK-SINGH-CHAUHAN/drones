#!/usr/bin/env python3
"""
ANTIGRAVITY — Mission Manager Unit Tests
==========================================
Tests mission state machine, waypoint management, and safety integration.
"""

import unittest
import numpy as np
from enum import IntEnum


class MissionState(IntEnum):
    IDLE = 0
    PLANNING = 1
    NAVIGATING = 2
    REACHED_WAYPOINT = 3
    PAUSED = 4
    ABORTING = 5
    COMPLETED = 6
    FAILED = 7


class MissionStateMachine:
    """Minimal mission FSM for unit testing."""

    def __init__(self, goal_tolerance=0.5, waypoint_tolerance=1.0,
                 max_time=300.0):
        self.state = MissionState.IDLE
        self.goal_tolerance = goal_tolerance
        self.waypoint_tolerance = waypoint_tolerance
        self.max_time = max_time
        self.waypoints = []
        self.current_idx = 0
        self.elapsed = 0.0
        self.total_distance = 0.0
        self.safety_level = 'NOMINAL'

    def start_mission(self, waypoints):
        if not waypoints:
            return False
        self.waypoints = waypoints
        self.current_idx = 0
        self.total_distance = 0.0
        self.elapsed = 0.0
        self.state = MissionState.PLANNING
        return True

    def receive_path(self, path_length):
        if self.state == MissionState.PLANNING:
            if path_length > 0:
                self.state = MissionState.NAVIGATING
                return True
            return False
        return False

    def tick(self, current_pos, dt=1.0):
        """Advance mission state machine."""
        self.elapsed += dt

        if self.state == MissionState.IDLE:
            return

        if self.state == MissionState.PAUSED:
            return

        # Safety check
        if self.safety_level in ('CRITICAL', 'EMERGENCY'):
            if self.state == MissionState.NAVIGATING:
                self.state = MissionState.PAUSED
            return

        # Timeout
        if self.elapsed > self.max_time:
            self.state = MissionState.FAILED
            return

        if self.state == MissionState.NAVIGATING:
            self._check_waypoint(current_pos)

    def _check_waypoint(self, current_pos):
        if self.current_idx >= len(self.waypoints):
            return

        target = np.array(self.waypoints[self.current_idx])
        dist = np.linalg.norm(np.array(current_pos) - target)

        is_final = self.current_idx == len(self.waypoints) - 1
        tolerance = self.goal_tolerance if is_final else self.waypoint_tolerance

        if dist < tolerance:
            self.current_idx += 1
            if self.current_idx >= len(self.waypoints):
                self.state = MissionState.COMPLETED
            else:
                self.state = MissionState.PLANNING

    def pause(self):
        if self.state == MissionState.NAVIGATING:
            self.state = MissionState.PAUSED

    def resume(self):
        if self.state == MissionState.PAUSED:
            self.state = MissionState.NAVIGATING

    def abort(self):
        self.state = MissionState.ABORTING


class TestMissionStart(unittest.TestCase):
    def test_start_with_waypoints(self):
        fsm = MissionStateMachine()
        result = fsm.start_mission([[5, 5, 1.5]])
        self.assertTrue(result)
        self.assertEqual(fsm.state, MissionState.PLANNING)

    def test_start_with_empty_waypoints(self):
        fsm = MissionStateMachine()
        result = fsm.start_mission([])
        self.assertFalse(result)
        self.assertEqual(fsm.state, MissionState.IDLE)

    def test_multi_waypoint_start(self):
        fsm = MissionStateMachine()
        result = fsm.start_mission([[1, 1, 1], [5, 5, 1], [10, 10, 1]])
        self.assertTrue(result)
        self.assertEqual(len(fsm.waypoints), 3)
        self.assertEqual(fsm.current_idx, 0)


class TestMissionNavigation(unittest.TestCase):
    def test_path_transitions_to_navigating(self):
        fsm = MissionStateMachine()
        fsm.start_mission([[5, 5, 1.5]])
        result = fsm.receive_path(10)
        self.assertTrue(result)
        self.assertEqual(fsm.state, MissionState.NAVIGATING)

    def test_empty_path_stays_planning(self):
        fsm = MissionStateMachine()
        fsm.start_mission([[5, 5, 1.5]])
        result = fsm.receive_path(0)
        self.assertFalse(result)
        self.assertEqual(fsm.state, MissionState.PLANNING)

    def test_waypoint_reached(self):
        fsm = MissionStateMachine(goal_tolerance=0.5)
        fsm.start_mission([[5, 5, 1.5]])
        fsm.receive_path(10)
        fsm.tick([4.8, 5.1, 1.5])  # Within tolerance
        self.assertEqual(fsm.state, MissionState.COMPLETED)

    def test_waypoint_not_reached(self):
        fsm = MissionStateMachine(goal_tolerance=0.5)
        fsm.start_mission([[5, 5, 1.5]])
        fsm.receive_path(10)
        fsm.tick([0, 0, 1.5])  # Far away
        self.assertEqual(fsm.state, MissionState.NAVIGATING)


class TestMultiWaypointMission(unittest.TestCase):
    def test_advance_through_waypoints(self):
        fsm = MissionStateMachine(goal_tolerance=0.5, waypoint_tolerance=1.0)
        fsm.start_mission([[3, 0, 1], [6, 0, 1], [10, 0, 1]])
        fsm.receive_path(10)

        # Reach first waypoint
        fsm.tick([3.0, 0.0, 1.0])
        self.assertEqual(fsm.state, MissionState.PLANNING)
        self.assertEqual(fsm.current_idx, 1)

        # Receive path and navigate to second
        fsm.receive_path(5)
        fsm.tick([6.0, 0.0, 1.0])
        self.assertEqual(fsm.state, MissionState.PLANNING)
        self.assertEqual(fsm.current_idx, 2)

        # Navigate to final
        fsm.receive_path(3)
        fsm.tick([10.0, 0.0, 1.0])
        self.assertEqual(fsm.state, MissionState.COMPLETED)

    def test_final_waypoint_uses_tighter_tolerance(self):
        fsm = MissionStateMachine(goal_tolerance=0.3, waypoint_tolerance=2.0)
        fsm.start_mission([[5, 0, 1], [10, 0, 1]])

        # First WP uses loose tolerance
        fsm.receive_path(10)
        fsm.tick([3.5, 0, 1])  # Within 2.0m of WP1 (5,0,1)
        self.assertEqual(fsm.state, MissionState.PLANNING)

        # Final WP uses tight tolerance
        fsm.receive_path(5)
        fsm.tick([9.5, 0, 1])  # 0.5m away — outside 0.3m tolerance
        self.assertEqual(fsm.state, MissionState.NAVIGATING)

        fsm.tick([9.9, 0, 1])  # 0.1m — within 0.3m
        self.assertEqual(fsm.state, MissionState.COMPLETED)


class TestMissionSafety(unittest.TestCase):
    def test_critical_pauses_mission(self):
        fsm = MissionStateMachine()
        fsm.start_mission([[5, 5, 1.5]])
        fsm.receive_path(10)
        fsm.safety_level = 'CRITICAL'
        fsm.tick([0, 0, 1.5])
        self.assertEqual(fsm.state, MissionState.PAUSED)

    def test_emergency_pauses_mission(self):
        fsm = MissionStateMachine()
        fsm.start_mission([[5, 5, 1.5]])
        fsm.receive_path(10)
        fsm.safety_level = 'EMERGENCY'
        fsm.tick([0, 0, 1.5])
        self.assertEqual(fsm.state, MissionState.PAUSED)

    def test_nominal_continues_navigation(self):
        fsm = MissionStateMachine()
        fsm.start_mission([[5, 5, 1.5]])
        fsm.receive_path(10)
        fsm.safety_level = 'NOMINAL'
        fsm.tick([0, 0, 1.5])
        self.assertEqual(fsm.state, MissionState.NAVIGATING)


class TestMissionCommands(unittest.TestCase):
    def test_pause_resume(self):
        fsm = MissionStateMachine()
        fsm.start_mission([[5, 5, 1.5]])
        fsm.receive_path(10)

        fsm.pause()
        self.assertEqual(fsm.state, MissionState.PAUSED)

        fsm.resume()
        self.assertEqual(fsm.state, MissionState.NAVIGATING)

    def test_abort(self):
        fsm = MissionStateMachine()
        fsm.start_mission([[5, 5, 1.5]])
        fsm.receive_path(10)
        fsm.abort()
        self.assertEqual(fsm.state, MissionState.ABORTING)

    def test_pause_only_when_navigating(self):
        fsm = MissionStateMachine()
        fsm.pause()  # IDLE — should not change
        self.assertEqual(fsm.state, MissionState.IDLE)

    def test_resume_only_when_paused(self):
        fsm = MissionStateMachine()
        fsm.start_mission([[5, 5, 1.5]])
        fsm.receive_path(10)
        fsm.resume()  # Already navigating — no change
        self.assertEqual(fsm.state, MissionState.NAVIGATING)


class TestMissionTimeout(unittest.TestCase):
    def test_timeout_fails_mission(self):
        fsm = MissionStateMachine(max_time=10.0)
        fsm.start_mission([[5, 5, 1.5]])
        fsm.receive_path(10)

        # Accumulate time past timeout
        for _ in range(12):
            fsm.tick([0, 0, 1.5], dt=1.0)

        self.assertEqual(fsm.state, MissionState.FAILED)

    def test_completion_before_timeout(self):
        fsm = MissionStateMachine(max_time=10.0, goal_tolerance=1.0)
        fsm.start_mission([[1, 0, 1.5]])
        fsm.receive_path(5)
        fsm.tick([1, 0, 1.5], dt=1.0)  # Reached immediately
        self.assertEqual(fsm.state, MissionState.COMPLETED)


if __name__ == '__main__':
    unittest.main()
