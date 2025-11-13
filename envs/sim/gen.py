"""
Synthetic episode generator for RL-STIR.
Generates realistic attack scenarios with tamper variants for training.

This module creates synthetic cybersecurity episodes that simulate real-world
attack scenarios. Each episode includes:
- Log records from various sources (Sysmon, audit logs, etc.)
- Process, file, and network graph structures
- Ground truth labels for TTPs and tampering
- Realistic timing and relationships between entities
"""

import json
import random
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

from data.schemas import (
    LogRecord, LogChannel, ProcNode, FileNode, SocketNode, GraphEdge, EdgeType,
    EpisodeMetadata, TTPSequence, TamperSpan, create_log_dataframe,
    create_proc_nodes_dataframe, create_file_nodes_dataframe,
    create_socket_nodes_dataframe, create_edges_dataframe
)

try:
    from .llm_helper import LLMContentGenerator
except ImportError:
    # Fallback if llm_helper is not available
    from llm_helper import LLMContentGenerator


@dataclass
class AttackScenario:
    """
    Attack scenario configuration
    
    Attributes:
        name: Unique identifier for the scenario
        ttp_sequence: List of MITRE ATT&CK technique IDs in execution order
        duration_minutes: Typical duration of this attack scenario
        tamper_probability: Probability that evidence tampering occurs
        complexity: Attack complexity level (simple, medium, complex)
        description: Human-readable description of the attack
    """
    name: str
    ttp_sequence: List[str]
    duration_minutes: int
    tamper_probability: float = 0.3
    complexity: str = "medium"
    description: Optional[str] = None


class SyntheticEpisodeGenerator:
    """
    Generates synthetic episodes with attack scenarios and tamper variants.
    
    This class creates realistic cybersecurity episodes by:
    1. Generating background system activity
    2. Injecting attack-specific behaviors based on MITRE ATT&CK techniques
    3. Optionally applying evidence tampering
    4. Creating graph structures representing process/file/network relationships
    """
    
    def __init__(self, output_dir: str = "data/episodes", use_llm: bool = False, 
                 llm_api_key: Optional[str] = None, benign_activity_ratio: float = 0.85):
        """
        Initialize the episode generator
        
        Args:
            output_dir: Directory to save generated episodes
            use_llm: Whether to use LLM for generating realistic content
            llm_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            benign_activity_ratio: Ratio of benign to malicious activity (0.0-1.0)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.benign_activity_ratio = benign_activity_ratio
        
        # Initialize LLM helper
        self.llm = LLMContentGenerator(use_llm=use_llm, api_key=llm_api_key)
        
        # Attack scenarios - expanded with more techniques and complexity
        self.scenarios = self._initialize_scenarios()
        
        # Common executables and processes
        self.benign_processes = [
            "chrome.exe", "firefox.exe", "notepad.exe", "explorer.exe",
            "svchost.exe", "winlogon.exe", "csrss.exe", "lsass.exe"
        ]
        
        self.suspicious_processes = [
            "powershell.exe", "cmd.exe", "wscript.exe", "cscript.exe",
            "rundll32.exe", "regsvr32.exe", "certutil.exe", "bitsadmin.exe"
        ]
        
        # Common file paths
        self.temp_paths = [
            "C:\\Windows\\Temp\\", "C:\\Users\\Public\\", 
            "C:\\ProgramData\\", "C:\\Windows\\System32\\"
        ]
        
        # Network destinations
        self.suspicious_ips = [
            "192.168.1.100", "10.0.0.50", "172.16.0.25"
        ]
        
        self.suspicious_ports = [4444, 8080, 9999, 1337]
    
    def _initialize_scenarios(self) -> Dict[str, AttackScenario]:
        """Initialize all attack scenarios with expanded TTP coverage"""
        scenarios = {
            # Original scenarios - expanded
            "phishing_script_lolbin": AttackScenario(
                name="phishing_script_lolbin",
                ttp_sequence=["T1566.001", "T1204.002", "T1059.001", "T1218.011", "T1071.001", "T1055.001"],
                duration_minutes=45,
                tamper_probability=0.4,
                complexity="medium",
                description="Phishing email leads to script execution, LOLBin usage, and C2 communication"
            ),
            "ssh_brute_force": AttackScenario(
                name="ssh_brute_force", 
                ttp_sequence=["T1110.001", "T1078.003", "T1021.001", "T1071.001", "T1083.001"],
                duration_minutes=30,
                tamper_probability=0.2,
                complexity="simple",
                description="SSH brute force attack followed by lateral movement"
            ),
            "crypto_miner": AttackScenario(
                name="crypto_miner",
                ttp_sequence=["T1059.001", "T1053.003", "T1071.001", "T1496", "T1055.001"],
                duration_minutes=60,
                tamper_probability=0.3,
                complexity="medium",
                description="Cryptocurrency mining malware with persistence mechanisms"
            ),
            "ransomware": AttackScenario(
                name="ransomware",
                ttp_sequence=["T1059.001", "T1486", "T1071.001", "T1489", "T1490"],
                duration_minutes=90,
                tamper_probability=0.5,
                complexity="complex",
                description="Ransomware attack with data encryption and service disruption"
            ),
            
            # New scenarios - lateral movement
            "lateral_movement_apt": AttackScenario(
                name="lateral_movement_apt",
                ttp_sequence=["T1078.001", "T1021.001", "T1021.002", "T1055.001", "T1083.001", "T1005", "T1041"],
                duration_minutes=120,
                tamper_probability=0.3,
                complexity="complex",
                description="Advanced persistent threat with lateral movement and data collection"
            ),
            
            # New scenarios - data exfiltration
            "data_exfiltration": AttackScenario(
                name="data_exfiltration",
                ttp_sequence=["T1078.002", "T1083.001", "T1005", "T1020", "T1041", "T1071.001"],
                duration_minutes=180,
                tamper_probability=0.4,
                complexity="complex",
                description="Stolen credentials used to access and exfiltrate sensitive data"
            ),
            
            # New scenarios - persistence
            "persistence_backdoor": AttackScenario(
                name="persistence_backdoor",
                ttp_sequence=["T1059.001", "T1053.003", "T1543.003", "T1547.001", "T1071.001"],
                duration_minutes=60,
                tamper_probability=0.3,
                complexity="medium",
                description="Backdoor installation with multiple persistence mechanisms"
            ),
            
            # New scenarios - privilege escalation
            "privilege_escalation": AttackScenario(
                name="privilege_escalation",
                ttp_sequence=["T1078.003", "T1055.001", "T1547.001", "T1548.002", "T1083.001"],
                duration_minutes=45,
                tamper_probability=0.2,
                complexity="medium",
                description="Privilege escalation through process injection and UAC bypass"
            ),
            
            # New scenarios - defense evasion
            "defense_evasion": AttackScenario(
                name="defense_evasion",
                ttp_sequence=["T1059.001", "T1055.001", "T1070.001", "T1070.004", "T1562.001", "T1071.001"],
                duration_minutes=75,
                tamper_probability=0.6,
                complexity="complex",
                description="Attack with multiple defense evasion techniques including log deletion"
            ),
            
            # New scenarios - credential access
            "credential_dumping": AttackScenario(
                name="credential_dumping",
                ttp_sequence=["T1078.001", "T1055.001", "T1003.001", "T1003.002", "T1071.001"],
                duration_minutes=30,
                tamper_probability=0.4,
                complexity="medium",
                description="Credential dumping attack using process injection techniques"
            )
        }
        
        # Generate descriptions using LLM if available
        for scenario in scenarios.values():
            if not scenario.description:
                scenario.description = self.llm.generate_attack_description(scenario.ttp_sequence)
        
        return scenarios

    def generate_episode(self, scenario_name: str, case_id: Optional[str] = None) -> str:
        """Generate a complete episode with logs and graph data"""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
            
        scenario = self.scenarios[scenario_name]
        case_id = case_id or f"{scenario_name}_{int(time.time())}"
        
        # Generate base episode
        logs_df, proc_nodes, file_nodes, socket_nodes, edges = self._generate_base_episode(
            scenario, case_id
        )
        
        # Apply tampering if configured
        if random.random() < scenario.tamper_probability:
            logs_df = self._apply_tampering(logs_df, scenario)
        
        # Save episode data
        episode_dir = self.output_dir / case_id
        episode_dir.mkdir(exist_ok=True)
        
        # Save parquet files
        logs_df.to_parquet(episode_dir / "logs.parquet", index=False)
        proc_nodes.to_parquet(episode_dir / "proc_nodes.parquet", index=False)
        file_nodes.to_parquet(episode_dir / "file_nodes.parquet", index=False)
        socket_nodes.to_parquet(episode_dir / "sock_nodes.parquet", index=False)
        edges.to_parquet(episode_dir / "edges.parquet", index=False)
        
        # Save metadata and ground truth
        self._save_metadata(episode_dir, case_id, scenario, logs_df, len(proc_nodes))
        self._save_ground_truth(episode_dir, case_id, scenario)
        
        return str(episode_dir)

    def _generate_base_episode(self, scenario: AttackScenario, case_id: str) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
    ]:
        """Generate the base episode without tampering"""
        start_time = int(time.time() * 1e9)  # nanoseconds
        duration_ns = scenario.duration_minutes * 60 * 1e9
        
        logs_df = create_log_dataframe()
        proc_nodes = create_proc_nodes_dataframe()
        file_nodes = create_file_nodes_dataframe()
        socket_nodes = create_socket_nodes_dataframe()
        edges = create_edges_dataframe()
        
        # Generate background activity
        self._generate_background_activity(
            logs_df, proc_nodes, file_nodes, socket_nodes, edges,
            start_time, duration_ns
        )
        
        # Generate attack activity
        self._generate_attack_activity(
            logs_df, proc_nodes, file_nodes, socket_nodes, edges,
            scenario, start_time, duration_ns
        )
        
        return logs_df, proc_nodes, file_nodes, socket_nodes, edges

    def _generate_background_activity(self, logs_df, proc_nodes, file_nodes, 
                                    socket_nodes, edges, start_time, duration_ns):
        """
        Generate realistic background system activity
        
        Creates normal system operations that would occur in a typical
        enterprise environment, including:
        - User application launches
        - System service operations
        - File system operations
        - Network connections to legitimate services
        """
        # Calculate number of background events based on duration and ratio
        total_events = int(duration_ns / 1e9) * 10  # ~10 events per second baseline
        num_background_events = int(total_events * self.benign_activity_ratio)
        num_background_events = max(500, min(num_background_events, 5000))  # Reasonable bounds
        
        # Create process tree structure for more realism
        process_tree = {}  # pid -> ppid mapping
        
        for i in range(num_background_events):
            ts = start_time + random.randint(0, int(duration_ns))
            
            # Generate realistic process name
            exe = self.llm.generate_process_name(is_suspicious=False)
            pid = random.randint(1000, 8000)
            
            # Create realistic parent-child relationships
            if i > 0 and random.random() < 0.3:  # 30% chance of child process
                parent_pid = random.choice(list(process_tree.keys()) if process_tree else [1000])
                ppid = parent_pid
            else:
                ppid = random.randint(100, 1000)  # System process
            
            process_tree[pid] = ppid
            
            # Generate realistic command line
            if exe == "chrome.exe" or exe == "firefox.exe":
                cmdline = f"{exe} --new-window https://www.example.com"
            elif exe == "notepad.exe":
                file_path = self.llm.generate_file_path("user", {"user": "user1"})
                cmdline = f"{exe} {file_path}"
            else:
                cmdline = f"{exe}"
            
            # Create process node
            proc_node = {
                'pid': pid,
                'first_ts': ts,
                'last_ts': ts + random.randint(int(1e9), int(10e9)),  # 1-10 seconds
                'exe_hash': f"hash_{exe}_{pid}",
                'signed': True,
                'rare_parent': False,
                'user': random.choice(['user1', 'user2', 'admin', 'service_account']),
                'cmdline': cmdline
            }
            proc_nodes.loc[len(proc_nodes)] = proc_node
            
            # Create log entry with realistic event IDs
            event_id = random.choice([1, 3, 5, 7, 10, 11, 12, 13])  # Common Sysmon events
            
            log_entry = {
                'host': random.choice(['workstation-01', 'workstation-02', 'server-01', 'server-02']),
                'ts_ns': ts,
                'channel': random.choice(['sysmon', 'auditd', 'security']),
                'template_id': random.randint(1, 100),
                'pid': int(pid),
                'ppid': int(ppid),
                'tid': int(random.randint(1, 100)),
                'exe': exe,
                'cmdline': cmdline,
                'user': proc_node['user'],
                'sid': f"S-1-5-21-{random.randint(1000000000, 9999999999)}",
                'event_id': int(event_id),
                'fields_json': json.dumps({'normal': 'activity', 'event_type': 'process_create'}),
                'label_ttp': None,
                'tamper_tag': False
            }
            logs_df.loc[len(logs_df)] = log_entry
            
            # Occasionally create file operations
            if random.random() < 0.1:  # 10% chance
                file_path = self.llm.generate_file_path("user", {"user": proc_node['user']})
                file_node = {
                    'path': file_path,
                    'sha256': f"hash_{random.randint(100000, 999999)}",
                    'entropy': random.uniform(3.0, 6.0),
                    'size': random.randint(1024, 102400),
                    'created_ts': ts,
                    'modified_ts': ts + random.randint(int(1e8), int(1e9))
                }
                file_nodes.loc[len(file_nodes)] = file_node
                
                # Create file access edge
                edge = {
                    'src_id': f"proc_{pid}",
                    'dst_id': f"file_{len(file_nodes)-1}",
                    'etype': 'PROC_FILE_RW',
                    'weight': 1.0,
                    'first_ts': ts,
                    'last_ts': ts + random.randint(int(1e8), int(1e9))
                }
                edges.loc[len(edges)] = edge

    def _generate_attack_activity(self, logs_df, proc_nodes, file_nodes,
                                socket_nodes, edges, scenario, start_time, duration_ns):
        """Generate attack-specific activity based on scenario"""
        attack_start = start_time + random.randint(0, int(duration_ns * 0.3))
        
        # Route to appropriate attack generator
        if scenario.name == "phishing_script_lolbin":
            self._generate_phishing_attack(logs_df, proc_nodes, file_nodes, 
                                         socket_nodes, edges, attack_start, duration_ns)
        elif scenario.name == "ssh_brute_force":
            self._generate_ssh_brute_attack(logs_df, proc_nodes, socket_nodes,
                                          edges, attack_start, duration_ns)
        elif scenario.name == "crypto_miner":
            self._generate_miner_attack(logs_df, proc_nodes, socket_nodes,
                                      edges, attack_start, duration_ns)
        elif scenario.name == "ransomware":
            self._generate_ransomware_attack(logs_df, proc_nodes, file_nodes,
                                           edges, attack_start, duration_ns)
        elif scenario.name == "lateral_movement_apt":
            self._generate_lateral_movement_attack(logs_df, proc_nodes, file_nodes,
                                                  socket_nodes, edges, attack_start, duration_ns)
        elif scenario.name == "data_exfiltration":
            self._generate_data_exfiltration_attack(logs_df, proc_nodes, file_nodes,
                                                   socket_nodes, edges, attack_start, duration_ns)
        elif scenario.name == "persistence_backdoor":
            self._generate_persistence_attack(logs_df, proc_nodes, file_nodes,
                                             socket_nodes, edges, attack_start, duration_ns)
        elif scenario.name == "privilege_escalation":
            self._generate_privilege_escalation_attack(logs_df, proc_nodes, file_nodes,
                                                       socket_nodes, edges, attack_start, duration_ns)
        elif scenario.name == "defense_evasion":
            self._generate_defense_evasion_attack(logs_df, proc_nodes, file_nodes,
                                                  socket_nodes, edges, attack_start, duration_ns)
        elif scenario.name == "credential_dumping":
            self._generate_credential_dumping_attack(logs_df, proc_nodes, file_nodes,
                                                    socket_nodes, edges, attack_start, duration_ns)
        else:
            # Fallback: generate generic attack based on TTP sequence
            self._generate_generic_attack(logs_df, proc_nodes, file_nodes,
                                         socket_nodes, edges, scenario, attack_start, duration_ns)

    def _generate_phishing_attack(self, logs_df, proc_nodes, file_nodes, 
                                socket_nodes, edges, start_time, duration_ns):
        """
        Generate phishing -> script -> LOLBin -> C2 attack
        
        This simulates a common attack chain:
        1. Phishing email with malicious attachment (T1566.001)
        2. User opens attachment triggering script (T1204.002)
        3. PowerShell script execution (T1059.001)
        4. Living Off the Land Binary usage (T1218.011)
        5. Command and control communication (T1071.001)
        6. Process injection for evasion (T1055.001)
        """
        scenario = self.scenarios["phishing_script_lolbin"]
        ts = start_time
        
        # Phase 1: Phishing email download (T1566.001)
        download_file = self.llm.generate_file_path("user", {"user": "victim_user", "file": "invoice.pdf"})
        file_node = {
            'path': download_file,
            'sha256': f"malicious_hash_{random.randint(100000, 999999)}",
            'entropy': random.uniform(6.5, 7.5),  # High entropy for malicious file
            'size': random.randint(50000, 200000),
            'created_ts': ts,
            'modified_ts': ts + 1e9
        }
        file_nodes.loc[len(file_nodes)] = file_node
        
        # Phase 2: Script execution (T1204.002, T1059.001)
        exe = self.llm.generate_process_name(is_suspicious=True)
        pid = 5001
        cmdline = self.llm.generate_command_line("powershell", {
            "url": "http://malicious.com/payload.ps1",
            "path": download_file
        })
        
        proc_node = {
            'pid': pid,
            'first_ts': ts + 5e9,  # 5 seconds after download
            'last_ts': ts + 30e9,
            'exe_hash': f"hash_{exe}_{pid}",
            'signed': True,
            'rare_parent': True,
            'user': 'victim_user',
            'cmdline': cmdline
        }
        proc_nodes.loc[len(proc_nodes)] = proc_node
        
        # Phase 3: Generate logs for each TTP with realistic timing
        ttp_timestamps = np.linspace(ts + 5e9, ts + 30e9, len(scenario.ttp_sequence))
        
        for i, (ttp, ttp_ts) in enumerate(zip(scenario.ttp_sequence, ttp_timestamps)):
            # Generate context-specific command lines
            if ttp == "T1059.001":  # PowerShell
                cmdline = self.llm.generate_command_line("powershell", {"command": "download_and_execute"})
            elif ttp == "T1218.011":  # Rundll32
                cmdline = self.llm.generate_command_line("rundll32", {"script_path": download_file})
            else:
                cmdline = proc_node['cmdline']
            
            log_entry = {
                'host': 'workstation-01',
                'ts_ns': int(ttp_ts),
                'channel': 'sysmon',
                'template_id': 10 + i,
                'pid': int(pid),
                'ppid': int(1000),
                'tid': int(1),
                'exe': exe,
                'cmdline': cmdline,
                'user': 'victim_user',
                'sid': 'S-1-5-21-victim',
                'event_id': int(1),
                'fields_json': json.dumps({'ttp': ttp, 'suspicious': True, 'phase': i+1}),
                'label_ttp': ttp,
                'tamper_tag': False
            }
            logs_df.loc[len(logs_df)] = log_entry
        
        # Phase 4: C2 connection (T1071.001)
        c2_ip, c2_port = self.llm.generate_network_destination("c2")
        socket_node = {
            'l4_proto': 'TCP',
            'src_ip': '192.168.1.50',
            'dst_ip': c2_ip,
            'dst_port': c2_port,
            'asn': random.randint(10000, 99999),
            'geo': 'Unknown'
        }
        socket_nodes.loc[len(socket_nodes)] = socket_node
        
        # Create edges
        edge = {
            'src_id': f"proc_{pid}",
            'dst_id': f"sock_{len(socket_nodes)-1}",
            'etype': 'PROC_SOCK_CONN',
            'weight': 1.0,
            'first_ts': ts + 25e9,
            'last_ts': ts + 30e9
        }
        edges.loc[len(edges)] = edge
        
        # File access edge
        file_edge = {
            'src_id': f"proc_{pid}",
            'dst_id': f"file_{len(file_nodes)-1}",
            'etype': 'PROC_FILE_RW',
            'weight': 1.0,
            'first_ts': ts,
            'last_ts': ts + 5e9
        }
        edges.loc[len(edges)] = file_edge

    def _generate_ssh_brute_attack(self, logs_df, proc_nodes, socket_nodes, 
                                 edges, start_time, duration_ns):
        """Generate SSH brute force attack"""
        # Multiple failed SSH attempts
        for i in range(50):  # 50 failed attempts
            ts = start_time + i * 1e9  # 1 second between attempts
            
            log_entry = {
                'host': 'server-01',
                'ts_ns': ts,
                'channel': 'auth',
                'template_id': 20,
                'pid': None,
                'ppid': None,
                'tid': None,
                'exe': 'sshd',
                'cmdline': 'sshd: authentication failure',
                'user': 'root',
                'sid': None,
                'event_id': int(14),
                'fields_json': json.dumps({'auth_failure': True, 'attempt': i}),
                'label_ttp': 'T1110.001',
                'tamper_tag': False
            }
            logs_df.loc[len(logs_df)] = log_entry

    def _generate_miner_attack(self, logs_df, proc_nodes, socket_nodes, 
                             edges, start_time, duration_ns):
        """Generate crypto miner attack"""
        # Mining process
        ts = start_time
        pid = 6001
        exe = "xmrig.exe"
        
        proc_node = {
            'pid': pid,
            'first_ts': ts,
            'last_ts': ts + duration_ns,
            'exe_hash': f"hash_{exe}_{pid}",
            'signed': False,
            'rare_parent': True,
            'user': 'miner_user',
            'cmdline': f"{exe} --pool=pool.example.com:4444"
        }
        proc_nodes.loc[len(proc_nodes)] = proc_node
        
        # Mining activity logs
        for i in range(100):
            log_ts = ts + i * (duration_ns // 100)
            log_entry = {
                'host': 'workstation-01',
                'ts_ns': int(log_ts),
                'channel': 'sysmon',
                'template_id': 30,
                'pid': int(pid),
                'ppid': int(1000),
                'tid': int(1),
                'exe': exe,
                'cmdline': f"{exe} --pool=pool.example.com:4444",
                'user': 'miner_user',
                'sid': 'S-1-5-21-miner',
                'event_id': int(1),
                'fields_json': json.dumps({'mining': True, 'hashrate': random.randint(100, 1000)}),
                'label_ttp': 'T1496',
                'tamper_tag': False
            }
            logs_df.loc[len(logs_df)] = log_entry

    def _generate_ransomware_attack(self, logs_df, proc_nodes, file_nodes, 
                                  edges, start_time, duration_ns):
        """Generate ransomware attack"""
        # Ransomware process
        ts = start_time
        pid = 7001
        exe = "encrypt.exe"
        
        proc_node = {
            'pid': pid,
            'first_ts': ts,
            'last_ts': ts + duration_ns,
            'exe_hash': f"hash_{exe}_{pid}",
            'signed': False,
            'rare_parent': True,
            'user': 'admin',
            'cmdline': f"{exe} --encrypt --path=C:\\Users"
        }
        proc_nodes.loc[len(proc_nodes)] = proc_node
        
        # File encryption activity
        for i in range(200):
            file_ts = ts + i * (duration_ns // 200)
            
            # Create file node
            file_path = f"C:\\Users\\Documents\\file_{i}.txt"
            file_node = {
                'path': file_path,
                'sha256': f"encrypted_hash_{i}",
                'entropy': 7.5,  # High entropy for encrypted files
                'size': random.randint(1024, 10240),
                'created_ts': file_ts,
                'modified_ts': file_ts + 1e9
            }
            file_nodes.loc[len(file_nodes)] = file_node
            
            # Create log entry
            log_entry = {
                'host': 'workstation-01',
                'ts_ns': int(file_ts),
                'channel': 'sysmon',
                'template_id': 40,
                'pid': int(pid),
                'ppid': int(1000),
                'tid': int(1),
                'exe': exe,
                'cmdline': f"{exe} --encrypt --path={file_path}",
                'user': 'admin',
                'sid': 'S-1-5-21-admin',
                'event_id': int(11),  # File create
                'fields_json': json.dumps({'file_encrypted': True, 'path': file_path}),
                'label_ttp': 'T1486',
                'tamper_tag': False
            }
            logs_df.loc[len(logs_df)] = log_entry
            
            # Create file access edge
            edge = {
                'src_id': f"proc_{pid}",
                'dst_id': f"file_{len(file_nodes)-1}",
                'etype': 'PROC_FILE_RW',
                'weight': 1.0,
                'first_ts': file_ts,
                'last_ts': file_ts + 1e9
            }
            edges.loc[len(edges)] = edge
    
    def _generate_lateral_movement_attack(self, logs_df, proc_nodes, file_nodes,
                                         socket_nodes, edges, start_time, duration_ns):
        """Generate lateral movement APT attack"""
        scenario = self.scenarios["lateral_movement_apt"]
        ts = start_time
        
        # Initial compromise (T1078.001)
        initial_pid = 8001
        exe = self.llm.generate_process_name(is_suspicious=True)
        cmdline = self.llm.generate_command_line("powershell", {"command": "lateral_movement"})
        
        proc_node = {
            'pid': initial_pid,
            'first_ts': ts,
            'last_ts': ts + duration_ns,
            'exe_hash': f"hash_{exe}_{initial_pid}",
            'signed': False,
            'rare_parent': True,
            'user': 'compromised_user',
            'cmdline': cmdline
        }
        proc_nodes.loc[len(proc_nodes)] = proc_node
        
        # Generate TTP sequence with realistic timing
        ttp_timestamps = np.linspace(ts, ts + duration_ns, len(scenario.ttp_sequence))
        for i, (ttp, ttp_ts) in enumerate(zip(scenario.ttp_sequence, ttp_timestamps)):
            log_entry = {
                'host': random.choice(['server-01', 'server-02', 'workstation-01']),
                'ts_ns': int(ttp_ts),
                'channel': 'sysmon',
                'template_id': 50 + i,
                'pid': int(initial_pid + i),
                'ppid': int(initial_pid if i > 0 else 1000),
                'tid': int(1),
                'exe': exe,
                'cmdline': cmdline,
                'user': 'compromised_user',
                'sid': 'S-1-5-21-compromised',
                'event_id': int(1),
                'fields_json': json.dumps({'ttp': ttp, 'lateral_movement': True}),
                'label_ttp': ttp,
                'tamper_tag': False
            }
            logs_df.loc[len(logs_df)] = log_entry
    
    def _generate_data_exfiltration_attack(self, logs_df, proc_nodes, file_nodes,
                                          socket_nodes, edges, start_time, duration_ns):
        """Generate data exfiltration attack"""
        scenario = self.scenarios["data_exfiltration"]
        ts = start_time
        
        # Credential theft and data access
        pid = 9001
        exe = self.llm.generate_process_name(is_suspicious=True)
        
        # Create multiple file accesses (T1083.001, T1005)
        for i in range(50):
            file_ts = ts + i * (duration_ns // 50)
            file_path = self.llm.generate_file_path("user", {"user": "target_user"})
            
            file_node = {
                'path': file_path,
                'sha256': f"data_hash_{random.randint(100000, 999999)}",
                'entropy': random.uniform(4.0, 7.0),
                'size': random.randint(10000, 1000000),
                'created_ts': file_ts,
                'modified_ts': file_ts + 1e9
            }
            file_nodes.loc[len(file_nodes)] = file_node
            
            log_entry = {
                'host': 'server-01',
                'ts_ns': int(file_ts),
                'channel': 'sysmon',
                'template_id': 60,
                'pid': int(pid),
                'ppid': int(1000),
                'tid': int(1),
                'exe': exe,
                'cmdline': f"{exe} --read {file_path}",
                'user': 'stolen_credential_user',
                'sid': 'S-1-5-21-stolen',
                'event_id': int(11),
                'fields_json': json.dumps({'ttp': 'T1005', 'data_access': True}),
                'label_ttp': 'T1005',
                'tamper_tag': False
            }
            logs_df.loc[len(logs_df)] = log_entry
        
        # Exfiltration (T1041)
        exfil_ip, exfil_port = self.llm.generate_network_destination("exfil")
        socket_node = {
            'l4_proto': 'TCP',
            'src_ip': '10.0.0.100',
            'dst_ip': exfil_ip,
            'dst_port': exfil_port,
            'asn': random.randint(10000, 99999),
            'geo': 'Unknown'
        }
        socket_nodes.loc[len(socket_nodes)] = socket_node
    
    def _generate_persistence_attack(self, logs_df, proc_nodes, file_nodes,
                                    socket_nodes, edges, start_time, duration_ns):
        """Generate persistence/backdoor attack"""
        scenario = self.scenarios["persistence_backdoor"]
        ts = start_time
        
        # Multiple persistence mechanisms
        pid = 10001
        exe = self.llm.generate_process_name(is_suspicious=True)
        
        ttp_timestamps = np.linspace(ts, ts + duration_ns, len(scenario.ttp_sequence))
        for i, (ttp, ttp_ts) in enumerate(zip(scenario.ttp_sequence, ttp_timestamps)):
            if ttp == "T1053.003":  # Scheduled task
                cmdline = self.llm.generate_command_line("schtasks", {"command": "create_backdoor"})
            elif ttp == "T1543.003":  # Service creation
                cmdline = f"sc.exe create BackdoorService binPath= {exe}"
            else:
                cmdline = self.llm.generate_command_line("powershell", {"command": "persistence"})
            
            log_entry = {
                'host': 'workstation-01',
                'ts_ns': int(ttp_ts),
                'channel': 'sysmon',
                'template_id': 70 + i,
                'pid': int(pid),
                'ppid': int(1000),
                'tid': int(1),
                'exe': exe,
                'cmdline': cmdline,
                'user': 'system',
                'sid': 'S-1-5-18',
                'event_id': int(1),
                'fields_json': json.dumps({'ttp': ttp, 'persistence': True}),
                'label_ttp': ttp,
                'tamper_tag': False
            }
            logs_df.loc[len(logs_df)] = log_entry
    
    def _generate_privilege_escalation_attack(self, logs_df, proc_nodes, file_nodes,
                                             socket_nodes, edges, start_time, duration_ns):
        """Generate privilege escalation attack"""
        scenario = self.scenarios["privilege_escalation"]
        ts = start_time
        
        # Process injection and UAC bypass
        pid = 11001
        exe = self.llm.generate_process_name(is_suspicious=True)
        
        ttp_timestamps = np.linspace(ts, ts + duration_ns, len(scenario.ttp_sequence))
        for i, (ttp, ttp_ts) in enumerate(zip(scenario.ttp_sequence, ttp_timestamps)):
            log_entry = {
                'host': 'workstation-01',
                'ts_ns': int(ttp_ts),
                'channel': 'sysmon',
                'template_id': 80 + i,
                'pid': int(pid + i),
                'ppid': int(1000),
                'tid': int(1),
                'exe': exe,
                'cmdline': self.llm.generate_command_line("powershell", {"command": "privilege_escalation"}),
                'user': 'low_priv_user' if i == 0 else 'SYSTEM',
                'sid': 'S-1-5-21-lowpriv' if i == 0 else 'S-1-5-18',
                'event_id': int(1),
                'fields_json': json.dumps({'ttp': ttp, 'privilege_escalation': True}),
                'label_ttp': ttp,
                'tamper_tag': False
            }
            logs_df.loc[len(logs_df)] = log_entry
    
    def _generate_defense_evasion_attack(self, logs_df, proc_nodes, file_nodes,
                                        socket_nodes, edges, start_time, duration_ns):
        """Generate defense evasion attack"""
        scenario = self.scenarios["defense_evasion"]
        ts = start_time
        
        # Multiple evasion techniques including log deletion
        pid = 12001
        exe = self.llm.generate_process_name(is_suspicious=True)
        
        ttp_timestamps = np.linspace(ts, ts + duration_ns, len(scenario.ttp_sequence))
        for i, (ttp, ttp_ts) in enumerate(zip(scenario.ttp_sequence, ttp_timestamps)):
            if ttp in ["T1070.001", "T1070.004"]:  # Log deletion
                cmdline = "wevtutil.exe cl Security"
            else:
                cmdline = self.llm.generate_command_line("powershell", {"command": "evasion"})
            
            log_entry = {
                'host': 'workstation-01',
                'ts_ns': int(ttp_ts),
                'channel': 'sysmon',
                'template_id': 90 + i,
                'pid': int(pid),
                'ppid': int(1000),
                'tid': int(1),
                'exe': exe,
                'cmdline': cmdline,
                'user': 'attacker',
                'sid': 'S-1-5-21-attacker',
                'event_id': int(1),
                'fields_json': json.dumps({'ttp': ttp, 'evasion': True}),
                'label_ttp': ttp,
                'tamper_tag': ttp in ["T1070.001", "T1070.004"]  # Mark log deletion as tampering
            }
            logs_df.loc[len(logs_df)] = log_entry
    
    def _generate_credential_dumping_attack(self, logs_df, proc_nodes, file_nodes,
                                            socket_nodes, edges, start_time, duration_ns):
        """Generate credential dumping attack"""
        scenario = self.scenarios["credential_dumping"]
        ts = start_time
        
        # Process injection and credential access
        pid = 13001
        exe = self.llm.generate_process_name(is_suspicious=True)
        
        ttp_timestamps = np.linspace(ts, ts + duration_ns, len(scenario.ttp_sequence))
        for i, (ttp, ttp_ts) in enumerate(zip(scenario.ttp_sequence, ttp_timestamps)):
            if ttp in ["T1003.001", "T1003.002"]:  # Credential dumping
                cmdline = "mimikatz.exe sekurlsa::logonpasswords"
            else:
                cmdline = self.llm.generate_command_line("powershell", {"command": "credential_access"})
            
            log_entry = {
                'host': 'server-01',
                'ts_ns': int(ttp_ts),
                'channel': 'sysmon',
                'template_id': 100 + i,
                'pid': int(pid),
                'ppid': int(1000),
                'tid': int(1),
                'exe': exe,
                'cmdline': cmdline,
                'user': 'SYSTEM',
                'sid': 'S-1-5-18',
                'event_id': int(1),
                'fields_json': json.dumps({'ttp': ttp, 'credential_dumping': True}),
                'label_ttp': ttp,
                'tamper_tag': False
            }
            logs_df.loc[len(logs_df)] = log_entry
    
    def _generate_generic_attack(self, logs_df, proc_nodes, file_nodes,
                                socket_nodes, edges, scenario, start_time, duration_ns):
        """Generate generic attack based on TTP sequence"""
        ts = start_time
        pid = 14001
        exe = self.llm.generate_process_name(is_suspicious=True)
        
        ttp_timestamps = np.linspace(ts, ts + duration_ns, len(scenario.ttp_sequence))
        for i, (ttp, ttp_ts) in enumerate(zip(scenario.ttp_sequence, ttp_timestamps)):
            log_entry = {
                'host': 'workstation-01',
                'ts_ns': int(ttp_ts),
                'channel': 'sysmon',
                'template_id': 200 + i,
                'pid': int(pid + i),
                'ppid': int(1000),
                'tid': int(1),
                'exe': exe,
                'cmdline': self.llm.generate_command_line("powershell", {"command": "attack"}),
                'user': 'attacker',
                'sid': 'S-1-5-21-attacker',
                'event_id': int(1),
                'fields_json': json.dumps({'ttp': ttp}),
                'label_ttp': ttp,
                'tamper_tag': False
            }
            logs_df.loc[len(logs_df)] = log_entry

    def _apply_tampering(self, logs_df: pd.DataFrame, scenario: AttackScenario) -> pd.DataFrame:
        """Apply tampering to the episode"""
        tamper_type = random.choice(['drop_events', 'jitter_timestamps', 'orphan_processes'])
        
        if tamper_type == 'drop_events':
            # Randomly drop 10-30% of events
            drop_ratio = random.uniform(0.1, 0.3)
            num_drop = int(len(logs_df) * drop_ratio)
            drop_indices = random.sample(range(len(logs_df)), num_drop)
            logs_df = logs_df.drop(drop_indices).reset_index(drop=True)
            
        elif tamper_type == 'jitter_timestamps':
            # Add random jitter to timestamps
            jitter_range = 1e9  # 1 second
            logs_df['ts_ns'] = logs_df['ts_ns'] + np.random.randint(
                -jitter_range, jitter_range, len(logs_df)
            )
            
        elif tamper_type == 'orphan_processes':
            # Remove parent process information
            orphan_mask = np.random.random(len(logs_df)) < 0.2
            logs_df.loc[orphan_mask, 'ppid'] = None
        
        # Mark tampered entries
        logs_df['tamper_tag'] = True
        
        return logs_df

    def _save_metadata(self, episode_dir: Path, case_id: str, scenario: AttackScenario,
                      logs_df: pd.DataFrame, num_proc_nodes: int):
        """Save episode metadata"""
        metadata = EpisodeMetadata(
            case_id=case_id,
            duration_ns=scenario.duration_minutes * 60 * 1e9,
            attack_type=scenario.name,
            tamper_applied=logs_df['tamper_tag'].any(),
            num_logs=len(logs_df),
            num_nodes=num_proc_nodes,
            num_edges=0  # Will be updated when edges are counted
        )
        
        with open(episode_dir / "metadata.json", 'w') as f:
            f.write(metadata.json())

    def _save_ground_truth(self, episode_dir: Path, case_id: str, scenario: AttackScenario):
        """Save ground truth TTP sequence and tamper spans"""
        # TTP sequence
        ttp_sequence = TTPSequence(
            case_id=case_id,
            ttp_sequence=scenario.ttp_sequence,
            timestamps=[int(time.time() * 1e9) + i * 1e9 for i in range(len(scenario.ttp_sequence))],
            confidence=[1.0] * len(scenario.ttp_sequence)
        )
        
        with open(episode_dir / "gt_ttp_sequence.json", 'w') as f:
            f.write(ttp_sequence.json())
        
        # Tamper spans (if any)
        tamper_spans = TamperSpan(
            case_id=case_id,
            tamper_spans=[],  # Will be populated if tampering is detected
            tamper_type=[]
        )
        
        with open(episode_dir / "gt_tamper_spans.json", 'w') as f:
            f.write(tamper_spans.json())


def main():
    """Generate sample episodes for testing"""
    generator = SyntheticEpisodeGenerator()
    
    # Generate one episode of each type
    for scenario_name in generator.scenarios.keys():
        print(f"Generating {scenario_name} episode...")
        episode_path = generator.generate_episode(scenario_name)
        print(f"Generated episode at: {episode_path}")


if __name__ == "__main__":
    main()
