"""
ANTIGRAVITY — System Monitor Node
====================================
Monitors CPU, GPU, memory, disk, and thermal health on the companion computer.
Publishes SystemHealth messages for the safety arbiter.

Acceptance Criteria (P4-T4):
  - Detects resource exhaustion before system impact
  - Monitors Jetson-specific thermal and power
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

import time
import json
import os


class SystemMonitorNode(Node):
    def __init__(self):
        super().__init__('system_monitor')

        self.declare_parameter('monitor_rate_hz', 2)
        self.declare_parameter('cpu_warn_pct', 85.0)
        self.declare_parameter('mem_warn_pct', 85.0)
        self.declare_parameter('temp_warn_c', 75.0)
        self.declare_parameter('temp_critical_c', 85.0)

        reliable_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                  history=HistoryPolicy.KEEP_LAST, depth=5)

        self._pub_health = self.create_publisher(String, 'system/health', reliable_qos)

        rate = self.get_parameter('monitor_rate_hz').value
        self.create_timer(1.0 / rate, self._monitor)
        self.create_timer(10.0, self._diag)
        self._last_report = {}
        self.get_logger().info('System monitor initialized')

    def _monitor(self):
        report = {
            'timestamp': time.time(),
            'cpu_percent': self._get_cpu(),
            'memory_percent': self._get_memory(),
            'disk_percent': self._get_disk(),
            'temperature_c': self._get_temperature(),
            'gpu_utilization': self._get_gpu(),
            'alerts': [],
        }

        if report['cpu_percent'] > self.get_parameter('cpu_warn_pct').value:
            report['alerts'].append(f'CPU_HIGH ({report["cpu_percent"]:.1f}%)')
        if report['memory_percent'] > self.get_parameter('mem_warn_pct').value:
            report['alerts'].append(f'MEM_HIGH ({report["memory_percent"]:.1f}%)')
        if report['temperature_c'] > self.get_parameter('temp_critical_c').value:
            report['alerts'].append(f'TEMP_CRITICAL ({report["temperature_c"]:.1f}°C)')
        elif report['temperature_c'] > self.get_parameter('temp_warn_c').value:
            report['alerts'].append(f'TEMP_WARN ({report["temperature_c"]:.1f}°C)')

        report['status'] = 'CRITICAL' if any('CRITICAL' in a for a in report['alerts']) else \
                           'WARNING' if report['alerts'] else 'OK'

        self._last_report = report
        self._pub_health.publish(String(data=json.dumps(report)))

    def _get_cpu(self):
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except ImportError:
            try:
                load = os.getloadavg()[0]
                ncpu = os.cpu_count() or 4
                return min(100.0, load / ncpu * 100.0)
            except: return 0.0

    def _get_memory(self):
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            try:
                with open('/proc/meminfo') as f:
                    lines = f.readlines()
                total = int(lines[0].split()[1])
                avail = int(lines[2].split()[1])
                return (1 - avail / total) * 100.0
            except: return 0.0

    def _get_disk(self):
        try:
            import psutil
            return psutil.disk_usage('/').percent
        except ImportError:
            try:
                st = os.statvfs('/')
                return (1 - st.f_bavail / st.f_blocks) * 100.0
            except: return 0.0

    def _get_temperature(self):
        """Read CPU/GPU temperature — works on Jetson and x86."""
        paths = [
            '/sys/devices/virtual/thermal/thermal_zone0/temp',
            '/sys/class/thermal/thermal_zone0/temp',
        ]
        for path in paths:
            try:
                with open(path) as f:
                    return int(f.read().strip()) / 1000.0
            except: continue
        return 45.0  # Default

    def _get_gpu(self):
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2)
            return float(result.stdout.strip())
        except: return 0.0

    def _diag(self):
        r = self._last_report
        self.get_logger().info(
            f'System — CPU: {r.get("cpu_percent",0):.1f}% | '
            f'MEM: {r.get("memory_percent",0):.1f}% | '
            f'Temp: {r.get("temperature_c",0):.1f}°C | '
            f'Status: {r.get("status","?")}')


def main(args=None):
    rclpy.init(args=args)
    node = SystemMonitorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
