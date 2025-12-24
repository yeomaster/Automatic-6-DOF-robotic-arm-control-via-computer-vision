#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================
Doosan 로봇 통합 제어 시스템 - 완전판
============================================
교육 내용 100% 반영:
- 2일차: Move J/L, Spiral 경로
- 3일차: 그리퍼 제어, Socket 통신
- 4일차: ROS2 로봇 제어
- 5일차: RealSense 카메라
- 7일차: MediaPipe 손 추적, Aruco Marker, Camera Calibration

추가 기능:
- 웹 기반 실시간 모니터링 대시보드
- 같은 WiFi 내 어느 기기에서든 접속 가능
============================================
"""

import sys
import os
import multiprocessing as mp
from multiprocessing import Process, Value, Event
import time
import math
import signal
import argparse
import socket
import threading
import json
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import struct

import numpy as np

# ============================================
# 공유 설정값
# ============================================
SAFETY_DISTANCE_M = 0.20          # 안전 거리 (20cm)
SAFETY_CLEAR_DISTANCE_M = 0.25    # 안전 해제 거리 (25cm)
CHECK_INTERVAL = 0.01             # 안전 체크 주기 (10ms)

# 2D 픽셀 기반 거리 계산 (Calibration 폴백용)
PIXEL_TO_CM_RATIO = 0.18
SAFETY_DISTANCE_PIXELS = int(SAFETY_DISTANCE_M * 100 / PIXEL_TO_CM_RATIO)
SAFETY_CLEAR_PIXELS = int(SAFETY_CLEAR_DISTANCE_M * 100 / PIXEL_TO_CM_RATIO)

# 로봇 설정
ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
VEL = 50
ACC = 50
DRAW_VEL = 20

# 원기둥 출력 설정
CYLINDER_RADIUS = 25
Z_PER_TURN = 1.2
NUMBER_OF_TURNS = 5
POINTS_PER_TURN = 60

# 카메라 설정
CONF = 0.2
MIN_MASK_PIXELS = 300
MIN_ASPECT = 3.0
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 2.0
MODEL_PATH = "pen_seg_final.pt"

# ============================================
# [3일차] Socket 통신 설정
# ============================================
SOCKET_ENABLED = True
SOCKET_PORT = 9999                # TCP 소켓 포트
WEB_PORT = 8080                   # 웹 모니터링 포트

# ============================================
# [7일차] ArUco 작업 영역 설정
# ============================================
ARUCO_ENABLED = True
ARUCO_MARKER_ID = 0               # 4개 마커 모두 같은 ID
ARUCO_MARKER_SIZE = 0.05

# ============================================
# 작업 영역 감속 설정
# ============================================
SLOWDOWN_ENABLED = True
SLOWDOWN_FACTOR = 0.7             # 30% 감속

# ============================================
# [7일차] Camera Calibration 설정
# ============================================
CALIBRATION_ENABLED = True
DEFAULT_WORKING_DISTANCE = 0.6    # 기본 작업 거리 (60cm)


# ============================================
# [7일차 교육] Camera Calibration 클래스
# ============================================
class CameraCalibration:
    """
    7일차 교육 - Camera Calibration
    
    RealSense 카메라의 내부 파라미터(intrinsics)를 이용하여
    픽셀 좌표를 실제 3D 좌표로 변환
    
    카메라 행렬 (Camera Matrix):
    | fx  0  cx |
    |  0 fy  cy |
    |  0  0   1 |
    
    - fx, fy: 초점 거리 (focal length) in pixels
    - cx, cy: 주점 (principal point) - 이미지 중심
    """
    
    def __init__(self):
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.width = 0
        self.height = 0
        self.is_calibrated = False
        self.default_depth = DEFAULT_WORKING_DISTANCE
        
        # 왜곡 계수 (distortion coefficients)
        self.dist_coeffs = None
        
    def calibrate_from_realsense(self, profile):
        """
        RealSense 카메라에서 내부 파라미터 추출
        
        교육 내용:
        - intrinsics.fx, fy: 초점 거리
        - intrinsics.ppx, ppy: 주점 좌표
        """
        try:
            import pyrealsense2 as rs
            
            # 컬러 스트림의 intrinsics 가져오기
            color_stream = profile.get_stream(rs.stream.color)
            intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            self.fx = intrinsics.fx
            self.fy = intrinsics.fy
            self.cx = intrinsics.ppx
            self.cy = intrinsics.ppy
            self.width = intrinsics.width
            self.height = intrinsics.height
            
            # 왜곡 계수 (RealSense는 보통 왜곡 보정됨)
            self.dist_coeffs = np.array(intrinsics.coeffs)
            
            self.is_calibrated = True
            
            print(f"\n{'='*55}")
            print(f"  [7일차 교육] Camera Calibration 완료")
            print(f"{'='*55}")
            print(f"  해상도: {self.width} x {self.height}")
            print(f"  초점거리 (fx, fy): ({self.fx:.1f}, {self.fy:.1f}) pixels")
            print(f"  주점 (cx, cy): ({self.cx:.1f}, {self.cy:.1f})")
            print(f"  왜곡 모델: {intrinsics.model}")
            print(f"{'='*55}\n")
            
            return True
            
        except Exception as e:
            print(f"[CALIB] RealSense 캘리브레이션 실패: {e}")
            print("[CALIB] 기본값으로 수동 캘리브레이션 진행...")
            self._set_defaults()
            return False
    
    def _set_defaults(self, width=1280, height=720, fov_h=87):
        """
        수동 캘리브레이션 (RealSense 없을 때)
        
        FOV(Field of View)에서 초점거리 계산:
        fx = width / (2 * tan(fov_h / 2))
        """
        self.width = width
        self.height = height
        self.cx = width / 2
        self.cy = height / 2
        
        # FOV에서 초점거리 계산
        fov_rad = np.radians(fov_h)
        self.fx = width / (2 * np.tan(fov_rad / 2))
        self.fy = self.fx  # 정사각형 픽셀 가정
        
        self.dist_coeffs = np.zeros(5)
        self.is_calibrated = True
        
        print(f"[CALIB] 수동 캘리브레이션 완료 (FOV={fov_h}°)")
    
    def pixel_to_3d(self, u, v, depth_m=None):
        """
        픽셀 좌표 (u, v) → 카메라 좌표계 3D 좌표 (X, Y, Z)
        
        교육 내용 - 핀홀 카메라 모델:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth
        
        Args:
            u, v: 픽셀 좌표
            depth_m: 깊이 값 (미터), None이면 기본값 사용
            
        Returns:
            np.array([X, Y, Z]): 카메라 좌표계 3D 좌표 (미터)
        """
        if not self.is_calibrated:
            return None
        
        if depth_m is None:
            depth_m = self.default_depth
            
        X = (u - self.cx) * depth_m / self.fx
        Y = (v - self.cy) * depth_m / self.fy
        Z = depth_m
        
        return np.array([X, Y, Z])
    
    def calculate_3d_distance(self, point1_uv, point2_uv, depth1=None, depth2=None):
        """
        두 픽셀 좌표 간의 실제 3D 거리 계산
        
        [기존 방식] - 부정확 (깊이 무시)
        dist = sqrt((u1-u2)² + (v1-v2)²) * PIXEL_TO_CM_RATIO
        
        [캘리브레이션 방식] - 정확
        1. 각 픽셀을 3D 좌표로 변환
        2. 3D 유클리드 거리 계산
        
        Args:
            point1_uv, point2_uv: (u, v) 픽셀 좌표
            depth1, depth2: 각 점의 깊이 (미터)
            
        Returns:
            float: 두 점 사이의 실제 거리 (미터)
        """
        p1 = self.pixel_to_3d(point1_uv[0], point1_uv[1], depth1)
        p2 = self.pixel_to_3d(point2_uv[0], point2_uv[1], depth2)
        
        if p1 is None or p2 is None:
            # 폴백: 기존 2D 픽셀 방식
            pixel_dist = math.sqrt(
                (point1_uv[0] - point2_uv[0])**2 + 
                (point1_uv[1] - point2_uv[1])**2
            )
            return pixel_dist * PIXEL_TO_CM_RATIO / 100
        
        return np.linalg.norm(p1 - p2)
    
    def get_camera_matrix(self):
        """OpenCV용 카메라 행렬 반환"""
        if not self.is_calibrated:
            return None
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def get_info(self):
        """캘리브레이션 정보 딕셔너리 반환"""
        return {
            'is_calibrated': self.is_calibrated,
            'resolution': f"{self.width}x{self.height}",
            'fx': round(self.fx, 1) if self.fx else None,
            'fy': round(self.fy, 1) if self.fy else None,
            'cx': round(self.cx, 1) if self.cx else None,
            'cy': round(self.cy, 1) if self.cy else None,
        }


# ============================================
# [3일차 교육] Socket 모니터링 서버 + 웹서버
# ============================================
class SafetyMonitorServer:
    """
    3일차 교육 - Socket 통신
    
    기능:
    1. TCP 소켓: JSON 데이터 브로드캐스트 (telnet 접속 가능)
    2. HTTP 웹서버: 브라우저에서 실시간 모니터링
    
    사용법:
    - TCP: telnet [IP] 9999
    - 웹: http://[IP]:8080
    """
    
    def __init__(self, socket_port=9999, web_port=8080):
        self.socket_port = socket_port
        self.web_port = web_port
        
        # TCP 소켓 서버
        self.server_socket = None
        self.clients = []
        self.running = False
        
        # 상태 데이터
        self.current_status = {}
        self.log_history = []
        
        # 웹서버
        self.web_server = None
        self.web_thread = None
        
        # 콜백
        self.on_emergency_stop = None
        self.on_resume = None
        
    def start(self):
        """서버 시작"""
        self.running = True
        
        # 1. TCP 소켓 서버 시작
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.socket_port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)
            
            threading.Thread(target=self._accept_clients, daemon=True).start()
            self._log("SERVER", f"TCP 소켓 서버 시작 - 포트 {self.socket_port}")
        except Exception as e:
            print(f"[SOCKET] TCP 서버 시작 실패: {e}")
        
        # 2. 웹 서버 시작
        try:
            self._start_web_server()
        except Exception as e:
            print(f"[SOCKET] 웹 서버 시작 실패: {e}")
        
        ip = self._get_ip()
        print(f"\n{'='*60}")
        print(f"  [3일차 교육] Socket 모니터링 서버 활성화")
        print(f"{'='*60}")
        print(f"  📡 TCP 소켓: telnet {ip} {self.socket_port}")
        print(f"  🌐 웹 모니터링: http://{ip}:{self.web_port}")
        print(f"  📱 같은 WiFi 내 모든 기기에서 접속 가능!")
        print(f"{'='*60}\n")
        
        return True
    
    def _get_ip(self):
        """로컬 IP 주소 확인"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "localhost"
    
    def _start_web_server(self):
        """웹 서버 시작"""
        server = self
        
        class MonitorHandler(SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/' or self.path == '/index.html':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(server._get_html_page().encode('utf-8'))
                elif self.path == '/api/status':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(server.current_status, ensure_ascii=False).encode('utf-8'))
                elif self.path == '/api/logs':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(server.log_history[-50:], ensure_ascii=False).encode('utf-8'))
                else:
                    self.send_error(404)
            
            def log_message(self, format, *args):
                pass  # 로그 출력 비활성화
        
        self.web_server = HTTPServer(('0.0.0.0', self.web_port), MonitorHandler)
        self.web_thread = threading.Thread(target=self.web_server.serve_forever, daemon=True)
        self.web_thread.start()
    
    def _get_html_page(self):
        """실시간 모니터링 HTML 페이지"""
        return '''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 로봇 안전 모니터링</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s;
        }
        .card:hover { transform: translateY(-5px); }
        .card h2 {
            font-size: 1.2em;
            margin-bottom: 15px;
            color: #a0a0a0;
        }
        .status-box {
            text-align: center;
            padding: 30px;
            border-radius: 10px;
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .status-safe { background: linear-gradient(135deg, #00b894, #00cec9); }
        .status-caution { background: linear-gradient(135deg, #fdcb6e, #e17055); }
        .status-danger { background: linear-gradient(135deg, #d63031, #e74c3c); animation: pulse 0.5s infinite; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .metric:last-child { border-bottom: none; }
        .metric-label { color: #a0a0a0; }
        .metric-value { font-weight: bold; font-size: 1.2em; }
        .progress-bar {
            width: 100%;
            height: 30px;
            background: rgba(255,255,255,0.2);
            border-radius: 15px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00b894, #00cec9);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        .log-box {
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            padding: 15px;
            height: 200px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.85em;
        }
        .log-entry { padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .log-time { color: #74b9ff; }
        .log-type { color: #ffeaa7; margin: 0 10px; }
        .slowdown-badge {
            display: inline-block;
            background: #e17055;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            margin-top: 10px;
        }
        .workspace-indicator {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 25px;
            font-size: 1.1em;
            margin-top: 10px;
        }
        .ws-defined { background: #00b894; }
        .ws-undefined { background: #636e72; }
        .calibration-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.8em;
        }
        .calib-ok { background: #00b894; }
        .calib-fail { background: #d63031; }
        .refresh-info {
            text-align: center;
            color: #a0a0a0;
            font-size: 0.9em;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 로봇 안전 모니터링 대시보드</h1>
        
        <div class="grid">
            <!-- 안전 상태 -->
            <div class="card">
                <h2>📊 안전 상태</h2>
                <div id="status-box" class="status-box status-safe">SAFE</div>
                <div class="metric">
                    <span class="metric-label">손 감지</span>
                    <span id="hand-detected" class="metric-value">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">거리</span>
                    <span id="distance" class="metric-value">- cm</span>
                </div>
                <div class="metric">
                    <span class="metric-label">작업 영역</span>
                    <span id="workspace" class="metric-value">-</span>
                </div>
                <div id="slowdown-indicator"></div>
            </div>
            
            <!-- 로봇 상태 -->
            <div class="card">
                <h2>🦾 로봇 상태</h2>
                <div class="metric">
                    <span class="metric-label">상태</span>
                    <span id="robot-status" class="metric-value">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">현재 속도</span>
                    <span id="current-speed" class="metric-value">100%</span>
                </div>
                <h2 style="margin-top: 20px;">📈 작업 진행률</h2>
                <div class="progress-bar">
                    <div id="progress-fill" class="progress-fill" style="width: 0%;">0%</div>
                </div>
            </div>
            
            <!-- 캘리브레이션 정보 -->
            <div class="card">
                <h2>📷 카메라 캘리브레이션</h2>
                <div class="metric">
                    <span class="metric-label">상태</span>
                    <span id="calib-status" class="calibration-badge calib-ok">OK</span>
                </div>
                <div class="metric">
                    <span class="metric-label">해상도</span>
                    <span id="calib-resolution" class="metric-value">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">초점거리 (fx)</span>
                    <span id="calib-fx" class="metric-value">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">초점거리 (fy)</span>
                    <span id="calib-fy" class="metric-value">-</span>
                </div>
            </div>
            
            <!-- ArUco 마커 -->
            <div class="card">
                <h2>🎯 ArUco 작업 영역</h2>
                <div class="metric">
                    <span class="metric-label">마커 ID</span>
                    <span id="aruco-id" class="metric-value">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">감지된 마커</span>
                    <span id="aruco-count" class="metric-value">0/4</span>
                </div>
                <div class="metric">
                    <span class="metric-label">영역 정의</span>
                    <span id="workspace-defined" class="workspace-indicator ws-undefined">미정의</span>
                </div>
            </div>
        </div>
        
        <!-- 로그 -->
        <div class="card">
            <h2>📜 실시간 로그</h2>
            <div id="log-box" class="log-box"></div>
        </div>
        
        <div class="refresh-info">
            🔄 자동 새로고침: 100ms | 마지막 업데이트: <span id="last-update">-</span>
        </div>
    </div>
    
    <script>
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // 안전 상태
                    const statusBox = document.getElementById('status-box');
                    statusBox.textContent = data.status || 'SAFE';
                    statusBox.className = 'status-box status-' + (data.status || 'safe').toLowerCase();
                    
                    // 손 감지
                    document.getElementById('hand-detected').textContent = 
                        data.hand_detected ? '✅ 감지됨' : '❌ 미감지';
                    
                    // 거리
                    document.getElementById('distance').textContent = 
                        data.hand_distance_cm ? data.hand_distance_cm.toFixed(1) + ' cm' : '- cm';
                    
                    // 작업 영역
                    document.getElementById('workspace').textContent = 
                        data.in_workspace ? '🔴 영역 내' : '🟢 영역 외';
                    
                    // 감속 표시
                    const slowdownDiv = document.getElementById('slowdown-indicator');
                    if (data.is_slowdown) {
                        slowdownDiv.innerHTML = '<span class="slowdown-badge">🐢 감속 중 (70%)</span>';
                    } else {
                        slowdownDiv.innerHTML = '';
                    }
                    
                    // 로봇 상태
                    document.getElementById('robot-status').textContent = 
                        data.robot_paused ? '⏸️ 일시정지' : '▶️ 동작 중';
                    
                    // 속도
                    document.getElementById('current-speed').textContent = 
                        data.is_slowdown ? '70%' : '100%';
                    
                    // 진행률
                    const progress = data.progress || 0;
                    document.getElementById('progress-fill').style.width = progress + '%';
                    document.getElementById('progress-fill').textContent = progress.toFixed(1) + '%';
                    
                    // 캘리브레이션
                    if (data.calibration) {
                        const calib = data.calibration;
                        document.getElementById('calib-status').textContent = 
                            calib.is_calibrated ? 'OK' : 'FAIL';
                        document.getElementById('calib-status').className = 
                            'calibration-badge ' + (calib.is_calibrated ? 'calib-ok' : 'calib-fail');
                        document.getElementById('calib-resolution').textContent = calib.resolution || '-';
                        document.getElementById('calib-fx').textContent = calib.fx || '-';
                        document.getElementById('calib-fy').textContent = calib.fy || '-';
                    }
                    
                    // ArUco
                    document.getElementById('aruco-id').textContent = data.aruco_marker_id || '-';
                    document.getElementById('aruco-count').textContent = 
                        (data.aruco_detected || 0) + '/4';
                    
                    const wsDefEl = document.getElementById('workspace-defined');
                    if (data.workspace_defined) {
                        wsDefEl.textContent = '✅ 정의됨';
                        wsDefEl.className = 'workspace-indicator ws-defined';
                    } else {
                        wsDefEl.textContent = '❌ 미정의';
                        wsDefEl.className = 'workspace-indicator ws-undefined';
                    }
                    
                    // 업데이트 시간
                    document.getElementById('last-update').textContent = 
                        new Date().toLocaleTimeString();
                })
                .catch(err => console.error('Status fetch error:', err));
        }
        
        function updateLogs() {
            fetch('/api/logs')
                .then(response => response.json())
                .then(logs => {
                    const logBox = document.getElementById('log-box');
                    logBox.innerHTML = logs.slice().reverse().map(log => 
                        `<div class="log-entry">
                            <span class="log-time">${log.time}</span>
                            <span class="log-type">[${log.type}]</span>
                            <span>${log.message}</span>
                        </div>`
                    ).join('');
                })
                .catch(err => console.error('Log fetch error:', err));
        }
        
        // 100ms마다 상태 업데이트
        setInterval(updateStatus, 100);
        // 1초마다 로그 업데이트
        setInterval(updateLogs, 1000);
        
        // 초기 로드
        updateStatus();
        updateLogs();
    </script>
</body>
</html>'''
    
    def _accept_clients(self):
        """TCP 클라이언트 연결 수락"""
        while self.running:
            try:
                client, addr = self.server_socket.accept()
                self.clients.append(client)
                self._log("CONNECT", f"클라이언트 연결: {addr}")
                
                threading.Thread(target=self._handle_client, args=(client, addr), daemon=True).start()
                
                welcome = {"type": "welcome", "message": "Safety Monitor 연결됨"}
                self._send_to_client(client, welcome)
            except socket.timeout:
                continue
            except:
                break
    
    def _handle_client(self, client, addr):
        """TCP 클라이언트 명령 처리"""
        client.settimeout(0.5)
        
        while self.running:
            try:
                data = client.recv(1024)
                if not data:
                    break
                command = data.decode('utf-8').strip().upper()
                self._process_command(client, command)
            except socket.timeout:
                continue
            except:
                break
        
        if client in self.clients:
            self.clients.remove(client)
        self._log("DISCONNECT", f"연결 해제: {addr}")
        client.close()
    
    def _process_command(self, client, command):
        """TCP 명령 처리"""
        self._log("COMMAND", f"명령: {command}")
        
        if command == "STOP":
            if self.on_emergency_stop:
                self.on_emergency_stop()
            response = {"type": "response", "result": "비상 정지 실행"}
        elif command == "RESUME":
            if self.on_resume:
                self.on_resume()
            response = {"type": "response", "result": "작업 재개"}
        elif command == "STATUS":
            response = self.current_status
        elif command == "LOG":
            response = {"type": "log", "history": self.log_history[-20:]}
        else:
            response = {"type": "help", "commands": ["STOP", "RESUME", "STATUS", "LOG"]}
        
        self._send_to_client(client, response)
    
    def _send_to_client(self, client, data):
        """TCP 클라이언트에 전송"""
        try:
            message = json.dumps(data, ensure_ascii=False) + "\r\n"
            client.sendall(message.encode('utf-8'))
        except:
            pass
    
    def _log(self, event_type, message):
        """이벤트 로그"""
        log_entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": event_type,
            "message": message
        }
        self.log_history.append(log_entry)
        if len(self.log_history) > 100:
            self.log_history = self.log_history[-100:]
    
    def broadcast_status(self, status_data):
        """상태 브로드캐스트 (TCP + 웹)"""
        self.current_status = status_data
        self.current_status["timestamp"] = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # TCP 클라이언트들에게 전송
        dead_clients = []
        for client in self.clients:
            try:
                message = json.dumps(self.current_status, ensure_ascii=False) + "\r\n"
                client.sendall(message.encode('utf-8'))
            except:
                dead_clients.append(client)
        
        for client in dead_clients:
            if client in self.clients:
                self.clients.remove(client)
    
    def log_event(self, event_type, message):
        """외부에서 로그 추가"""
        self._log(event_type, message)
    
    def stop(self):
        """서버 종료"""
        self.running = False
        
        for client in self.clients:
            try:
                client.close()
            except:
                pass
        
        if self.server_socket:
            self.server_socket.close()
        
        if self.web_server:
            self.web_server.shutdown()
        
        print("[SOCKET] 서버 종료")


# ============================================
# ArUco 작업 영역 관리자 (같은 ID 4개 마커)
# ============================================
class WorkspaceManager:
    """
    같은 ID의 ArUco 마커 4개로 작업 영역 정의
    """
    
    def __init__(self, marker_id=0, marker_length=0.05):
        self.marker_id = marker_id
        self.marker_length = marker_length
        
        self.aruco_dict = None
        self.detector = None
        
        self.workspace_corners = []
        self.workspace_polygon = None
        self.is_workspace_defined = False
        
        self.detected_markers = []
        self.total_checks = 0
        self.in_workspace_count = 0
        
    def initialize(self):
        """ArUco 검출기 초기화"""
        try:
            import cv2
            
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            
            print(f"\n{'='*55}")
            print(f"  [7일차 교육] ArUco 작업 영역 감지 활성화")
            print(f"{'='*55}")
            print(f"  마커 ID: {self.marker_id} (4개 동일)")
            print(f"  마커 크기: {self.marker_length*100}cm")
            print(f"  감속 비율: {SLOWDOWN_FACTOR*100:.0f}%")
            print(f"{'='*55}\n")
            return True
        except Exception as e:
            print(f"[ARUCO] 초기화 실패: {e}")
            return False
    
    def _sort_corners_clockwise(self, points):
        """4개 점을 시계방향으로 정렬"""
        if len(points) != 4:
            return points
        
        points = np.array(points)
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]
        
        sums = sorted_points[:, 0] + sorted_points[:, 1]
        top_left_idx = np.argmin(sums)
        sorted_points = np.roll(sorted_points, -top_left_idx, axis=0)
        
        return sorted_points.tolist()
    
    def update(self, frame):
        """매 프레임 작업 영역 업데이트"""
        if self.detector is None:
            return False
        
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        self.detected_markers = []
        
        if ids is None:
            return self.is_workspace_defined
        
        ids_flat = ids.flatten()
        marker_centers = []
        
        for i, marker_id in enumerate(ids_flat):
            if marker_id == self.marker_id:
                marker_corners = corners[i][0]
                center = marker_corners.mean(axis=0)
                
                self.detected_markers.append({
                    'corners': marker_corners.astype(int).tolist(),
                    'center': (int(center[0]), int(center[1]))
                })
                marker_centers.append((int(center[0]), int(center[1])))
        
        if len(marker_centers) == 4:
            sorted_corners = self._sort_corners_clockwise(marker_centers)
            self.workspace_corners = sorted_corners
            self.workspace_polygon = np.array(sorted_corners, dtype=np.int32)
            self.is_workspace_defined = True
        
        return self.is_workspace_defined
    
    def is_point_in_workspace(self, point_uv):
        """점이 작업 영역 내에 있는지 확인"""
        if not self.is_workspace_defined:
            return False
        
        import cv2
        result = cv2.pointPolygonTest(
            self.workspace_polygon,
            (float(point_uv[0]), float(point_uv[1])),
            False
        )
        return result >= 0
    
    def check_hand(self, hand_uv):
        """손이 작업 영역 내인지 확인"""
        self.total_checks += 1
        
        in_workspace = self.is_point_in_workspace(hand_uv)
        
        if in_workspace:
            self.in_workspace_count += 1
            return True, "영역 내"
        
        return False, "영역 외"
    
    def draw_workspace(self, frame, hand_in_ws=False):
        """작업 영역 시각화"""
        import cv2
        vis = frame.copy()
        H, W = vis.shape[:2]
        
        marker_color = (0, 255, 255)
        
        for i, marker_info in enumerate(self.detected_markers):
            corners = np.array(marker_info['corners'], dtype=np.int32)
            center = marker_info['center']
            
            cv2.polylines(vis, [corners], True, marker_color, 3)
            
            for j, corner in enumerate(corners):
                color = (0, 0, 255) if j == 0 else marker_color
                cv2.circle(vis, tuple(corner), 5, color, -1)
            
            cv2.drawMarker(vis, center, (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
            cv2.putText(vis, f"#{i+1}", (center[0] + 10, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, marker_color, 2)
        
        if not self.is_workspace_defined:
            status_text = f"ArUco ID:{self.marker_id} - {len(self.detected_markers)}/4 markers"
            cv2.putText(vis, status_text, (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis
        
        if hand_in_ws:
            cv2.polylines(vis, [self.workspace_polygon], True, (0, 0, 255), 4)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [self.workspace_polygon], (0, 0, 255))
            cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
            cv2.putText(vis, "SLOWDOWN ZONE - HAND DETECTED", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.polylines(vis, [self.workspace_polygon], True, (0, 255, 0), 3)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [self.workspace_polygon], (0, 255, 0))
            cv2.addWeighted(overlay, 0.1, vis, 0.9, 0, vis)
            cv2.putText(vis, f"WORKSPACE OK | ID:{self.marker_id} x4", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis
    
    def get_info(self):
        """상태 정보 반환"""
        return {
            'marker_id': self.marker_id,
            'detected_count': len(self.detected_markers),
            'is_defined': self.is_workspace_defined,
            'total_checks': self.total_checks,
            'in_workspace_count': self.in_workspace_count
        }


# ============================================
# 의존성 체크 함수
# ============================================
def check_vision_dependencies():
    missing = []
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    try:
        import pyrealsense2
    except ImportError:
        missing.append("pyrealsense2")
    try:
        from ultralytics import YOLO
    except ImportError:
        missing.append("ultralytics")
    try:
        import mediapipe
    except ImportError:
        missing.append("mediapipe")
    return missing


def check_robot_dependencies():
    missing = []
    try:
        import rclpy
    except ImportError:
        missing.append("rclpy (ROS2)")
    try:
        import DR_init
    except ImportError:
        missing.append("DR_init (Doosan SDK)")
    return missing


# ============================================
# 비전 프로세스 (완전판)
# ============================================
def vision_process(
    stop_event: Event,
    camera_ready: Event,
    hand_detected: Value,
    hand_distance: Value,
    pen_x: Value,
    pen_y: Value,
    pen_z: Value,
    pen_pixel_x: Value,
    pen_pixel_y: Value,
    pen_detected: Value,
    hand_in_workspace: Value,
    is_slowdown: Value,
    # 캘리브레이션 정보 공유용
    calib_fx: Value,
    calib_fy: Value,
    calib_cx: Value,
    calib_cy: Value,
    calib_ok: Value,
    # ArUco 정보 공유용
    aruco_count: Value,
    workspace_defined: Value,
    simulate: bool = False
):
    """비전 프로세스 - Camera Calibration + ArUco + 손 추적"""
    if simulate:
        vision_process_simulate(
            stop_event, camera_ready, hand_detected, hand_distance,
            pen_x, pen_y, pen_z, pen_pixel_x, pen_pixel_y, pen_detected,
            hand_in_workspace, is_slowdown, calib_fx, calib_fy, calib_cx, calib_cy, calib_ok,
            aruco_count, workspace_defined
        )
        return

    import cv2
    import numpy as np
    import pyrealsense2 as rs
    from ultralytics import YOLO
    
    try:
        from mediapipe import solutions as mp_solutions
        mp_hands = mp_solutions.hands
        mp_draw = mp_solutions.drawing_utils
    except ImportError:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
    
    hands_tracker = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    def pick_best_pen_instance(res, masks_np, W, H):
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        best_i = None
        best_score = -1e18
        for i in range(len(cls_ids)):
            if cls_ids[i] != 0:
                continue
            m = masks_np[i] > 0.5
            if int(m.sum()) < MIN_MASK_PIXELS:
                continue
            ys, xs = np.where(m)
            if xs.size == 0:
                continue
            w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
            if max(w, h) / max(1, min(w, h)) < MIN_ASPECT:
                continue
            x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy().astype(float)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 1 or bh <= 1:
                continue
            score = float(confs[i]) * bw * bh - 0.001 * (
                (0.5 * (x1 + x2) - W / 2) ** 2 + (0.5 * (y1 + y2) - H / 2) ** 2
            )
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_i = i
        return best_i

    def mask_centroid_uv(mask_bool):
        ys, xs = np.where(mask_bool)
        if xs.size == 0:
            return None
        return int(np.round(xs.mean())), int(np.round(ys.mean()))

    # YOLO 모델 로드
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = YOLO(MODEL_PATH)
            print(f"[VISION] YOLO 모델 로드: {MODEL_PATH}")
        except Exception as e:
            print(f"[VISION] YOLO 로드 실패: {e}")

    # RealSense 파이프라인
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    
    USE_DEPTH = False
    try:
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        USE_DEPTH = True
    except:
        pass

    try:
        profile = pipeline.start(config)
    except Exception as e:
        print(f"[VISION] 카메라 시작 실패: {e}")
        return

    # ============================================
    # [7일차] Camera Calibration 초기화
    # ============================================
    calibration = CameraCalibration()
    if CALIBRATION_ENABLED:
        calibration.calibrate_from_realsense(profile)
    else:
        calibration._set_defaults()
    
    # 캘리브레이션 정보 공유
    if calibration.is_calibrated:
        calib_fx.value = calibration.fx
        calib_fy.value = calibration.fy
        calib_cx.value = calibration.cx
        calib_cy.value = calibration.cy
        calib_ok.value = 1
    
    # ============================================
    # [7일차] ArUco 작업 영역 초기화
    # ============================================
    workspace = None
    if ARUCO_ENABLED:
        workspace = WorkspaceManager(
            marker_id=ARUCO_MARKER_ID,
            marker_length=ARUCO_MARKER_SIZE
        )
        workspace.initialize()

    print(f"\n[VISION] === 시스템 설정 ===")
    print(f"[VISION] 안전 거리: {SAFETY_DISTANCE_M * 100:.0f}cm")
    print(f"[VISION] Calibration: {'✅' if calibration.is_calibrated else '❌'}")
    print(f"[VISION] ArUco: {'✅' if ARUCO_ENABLED else '❌'}")
    print(f"[VISION] 감속: {'✅' if SLOWDOWN_ENABLED else '❌'} ({SLOWDOWN_FACTOR*100:.0f}%)")

    print("[VISION] 카메라 워밍업...")
    for _ in range(30):
        pipeline.wait_for_frames()

    print("[VISION] ✅ 카메라 준비 완료!")
    camera_ready.set()

    try:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame() if USE_DEPTH else None

            if not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            H, W = color.shape[:2]

            vis = color.copy()
            pen_uv = None
            current_hand_in_ws = False

            # ArUco 업데이트
            if workspace is not None:
                workspace.update(color)
                aruco_count.value = len(workspace.detected_markers)
                workspace_defined.value = 1 if workspace.is_workspace_defined else 0

            # 펜 추적 (YOLO)
            if model is not None:
                res = model.predict(color, conf=CONF, verbose=False)[0]
                vis = res.plot()

                if res.masks is not None and len(res.masks.data) > 0:
                    masks = res.masks.data.cpu().numpy()
                    best_i = pick_best_pen_instance(res, masks, W, H)

                    if best_i is not None:
                        mask_big = cv2.resize(
                            masks[best_i], (W, H), interpolation=cv2.INTER_NEAREST
                        ) > 0.5
                        uv = mask_centroid_uv(mask_big)

                        if uv:
                            pen_uv = uv
                            pen_pixel_x.value = pen_uv[0]
                            pen_pixel_y.value = pen_uv[1]
                            pen_detected.value = 1
                            cv2.circle(vis, pen_uv, 10, (0, 255, 0), -1)

            if pen_uv is None:
                pen_uv = (pen_pixel_x.value, pen_pixel_y.value)
                pen_detected.value = 0
                cv2.circle(vis, pen_uv, 10, (0, 200, 0), 2)

            # 손 추적 (MediaPipe)
            min_distance = float('inf')
            min_hand_uv = None
            
            results_hand = hands_tracker.process(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))

            if results_hand.multi_hand_landmarks:
                for hand_landmarks in results_hand.multi_hand_landmarks:
                    mp_draw.draw_landmarks(vis, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    finger_tips = [4, 8, 12, 16, 20]
                    
                    for tip_idx in finger_tips:
                        lm = hand_landmarks.landmark[tip_idx]
                        hu, hv = int(lm.x * W), int(lm.y * H)
                        
                        if 0 <= hu < W and 0 <= hv < H:
                            # ============================================
                            # [7일차] Camera Calibration으로 정확한 거리 계산
                            # ============================================
                            hand_depth = None
                            pen_depth = None
                            
                            if USE_DEPTH and depth_frame:
                                try:
                                    hand_depth = depth_frame.get_distance(hu, hv)
                                    pen_depth = depth_frame.get_distance(pen_uv[0], pen_uv[1])
                                    
                                    if not (DEPTH_MIN_M < hand_depth < DEPTH_MAX_M):
                                        hand_depth = None
                                    if not (DEPTH_MIN_M < pen_depth < DEPTH_MAX_M):
                                        pen_depth = None
                                except:
                                    pass
                            
                            # 캘리브레이션된 3D 거리 계산
                            if calibration.is_calibrated:
                                distance_m = calibration.calculate_3d_distance(
                                    (hu, hv), pen_uv, hand_depth, pen_depth
                                )
                            else:
                                pixel_dist = math.sqrt((pen_uv[0] - hu)**2 + (pen_uv[1] - hv)**2)
                                distance_m = pixel_dist * PIXEL_TO_CM_RATIO / 100
                            
                            if distance_m < min_distance:
                                min_distance = distance_m
                                min_hand_uv = (hu, hv)
                    
                    # 작업 영역 체크 (손바닥 중앙)
                    lm_palm = hand_landmarks.landmark[9]
                    palm_u, palm_v = int(lm_palm.x * W), int(lm_palm.y * H)
                    
                    if workspace is not None and workspace.is_workspace_defined:
                        in_ws, _ = workspace.check_hand((palm_u, palm_v))
                        if in_ws:
                            current_hand_in_ws = True
                            cv2.circle(vis, (palm_u, palm_v), 15, (0, 165, 255), 3)

            # 상태 업데이트
            if min_distance < float('inf') and min_hand_uv is not None:
                hand_detected.value = 1
                hand_distance.value = min_distance
                hand_in_workspace.value = 1 if current_hand_in_ws else 0
                is_slowdown.value = 1 if (current_hand_in_ws and SLOWDOWN_ENABLED) else 0
                
                dist_cm = min_distance * 100
                
                if min_distance < SAFETY_DISTANCE_M:
                    color_line = (0, 0, 255)
                    status = "DANGER"
                elif min_distance < SAFETY_CLEAR_DISTANCE_M:
                    color_line = (0, 165, 255)
                    status = "CAUTION"
                else:
                    color_line = (0, 255, 0)
                    status = "SAFE"

                cv2.line(vis, pen_uv, min_hand_uv, color_line, 3)
                cv2.circle(vis, min_hand_uv, 12, color_line, 3)

                status_text = f"{status}: {dist_cm:.1f}cm"
                if is_slowdown.value == 1:
                    status_text += f" | SLOW {SLOWDOWN_FACTOR*100:.0f}%"
                cv2.putText(vis, status_text, (20, H - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_line, 2)
                
                if status == "DANGER":
                    cv2.rectangle(vis, (5, 5), (W-5, H-5), (0, 0, 255), 5)
            else:
                hand_detected.value = 0
                hand_distance.value = 999.0
                hand_in_workspace.value = 0
                is_slowdown.value = 0

            # 안전 거리 원
            cv2.circle(vis, pen_uv, int(SAFETY_DISTANCE_M * 100 / PIXEL_TO_CM_RATIO), (0, 0, 255), 1)
            cv2.circle(vis, pen_uv, int(SAFETY_CLEAR_DISTANCE_M * 100 / PIXEL_TO_CM_RATIO), (0, 165, 255), 1)

            # ArUco 시각화
            if workspace is not None:
                vis = workspace.draw_workspace(vis, hand_in_ws=current_hand_in_ws)

            # 캘리브레이션 정보 표시
            calib_text = f"Calib: {'OK' if calibration.is_calibrated else 'FAIL'}"
            cv2.putText(vis, calib_text, (W - 150, H - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Safety Monitor - Full System", vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    except Exception as e:
        print(f"[VISION] 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        hands_tracker.close()
        print("[VISION] 종료")


def vision_process_simulate(
    stop_event, camera_ready, hand_detected, hand_distance,
    pen_x, pen_y, pen_z, pen_pixel_x, pen_pixel_y, pen_detected,
    hand_in_workspace, is_slowdown, calib_fx, calib_fy, calib_cx, calib_cy, calib_ok,
    aruco_count, workspace_defined
):
    """시뮬레이션 모드"""
    import random
    
    print("[VISION-SIM] 시뮬레이션 시작")

    # 가짜 캘리브레이션 데이터
    calib_fx.value = 900.0
    calib_fy.value = 900.0
    calib_cx.value = 640.0
    calib_cy.value = 360.0
    calib_ok.value = 1
    
    pen_pixel_x.value = 640
    pen_pixel_y.value = 400

    time.sleep(1)
    camera_ready.set()
    print("[VISION-SIM] ✅ 준비 완료")

    # 랜덤하게 workspace 정의
    workspace_defined.value = 1
    aruco_count.value = 4

    try:
        import cv2
        import numpy as np

        while not stop_event.is_set():
            img = np.zeros((400, 600, 3), dtype=np.uint8)
            cv2.putText(img, "SIMULATION MODE", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            if random.random() < 0.3:
                hand_detected.value = 1
                hand_distance.value = random.uniform(0.10, 0.40)
                in_ws = random.random() < 0.5
                hand_in_workspace.value = 1 if in_ws else 0
                is_slowdown.value = 1 if (in_ws and SLOWDOWN_ENABLED) else 0
                
                dist_cm = hand_distance.value * 100
                status = "DANGER" if hand_distance.value < SAFETY_DISTANCE_M else "SAFE"
                cv2.putText(img, f"{status}: {dist_cm:.1f}cm", (20, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                hand_detected.value = 0
                hand_distance.value = 999.0
                hand_in_workspace.value = 0
                is_slowdown.value = 0

            cv2.imshow("SIM", img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                stop_event.set()
                break

    except:
        while not stop_event.is_set():
            time.sleep(0.1)
    finally:
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print("[VISION-SIM] 종료")


# ============================================
# 로봇 프로세스
# ============================================
def robot_process(
    stop_event: Event,
    camera_ready: Event,
    hand_detected: Value,
    hand_distance: Value,
    robot_paused: Value,
    current_progress: Value,
    hand_in_workspace: Value,
    is_slowdown: Value,
    simulate: bool = False
):
    """로봇 제어 프로세스"""
    if simulate:
        robot_process_simulate(
            stop_event, camera_ready, hand_detected, hand_distance,
            robot_paused, current_progress, hand_in_workspace, is_slowdown
        )
        return

    print("[ROBOT] 카메라 대기...")
    camera_ready.wait()
    print("[ROBOT] ✅ 로봇 초기화")

    import rclpy
    import DR_init
    
    GripperController = None
    for path in ['dsr_example.simple.gripper_drl_controller', 
                 'dsr_example2.simple.gripper_drl_controller', 
                 'gripper_drl_controller']:
        try:
            module = __import__(path, fromlist=['GripperController'])
            GripperController = module.GripperController
            break
        except:
            continue

    rclpy.init()
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    node = rclpy.create_node("dsr_control", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import movej, movel, posj, posx, set_robot_mode, ROBOT_MODE_AUTONOMOUS, wait
    
    try:
        from DSR_ROBOT2 import drl_script_stop
        HAS_STOP = True
    except:
        HAS_STOP = False

    set_robot_mode(ROBOT_MODE_AUTONOMOUS)

    gripper = None
    if GripperController:
        gripper = GripperController(node=node, namespace=ROBOT_ID)
        if gripper.initialize():
            wait(2)
        else:
            gripper = None

    JOINT_HOME = [0, 0, 90, 90, 90, 0]
    START_POS = [350, -100, 45, 86, 93, 89]

    def check_safety():
        if hand_detected.value == 1:
            return hand_distance.value >= SAFETY_DISTANCE_M
        return True

    def wait_safety_clear():
        while not stop_event.is_set():
            if hand_detected.value == 0 or hand_distance.value >= SAFETY_CLEAR_DISTANCE_M:
                return True
            time.sleep(CHECK_INTERVAL)
        return False

    def safe_movel(pos, vel, acc):
        while not stop_event.is_set():
            if check_safety():
                robot_paused.value = 0
                current_vel = vel * SLOWDOWN_FACTOR if (SLOWDOWN_ENABLED and is_slowdown.value == 1) else vel
                movel(pos, current_vel, acc)
                return True
            else:
                robot_paused.value = 1
                if HAS_STOP:
                    try:
                        drl_script_stop(0)
                    except:
                        pass
                if not wait_safety_clear():
                    return False
        return False

    try:
        movej(posj(*JOINT_HOME), VEL, ACC)
        movel(posx(*START_POS), VEL, ACC)

        if gripper:
            gripper.move(643)
            time.sleep(0.5)

        cx, cy, z, rx, ry, rz = START_POS
        total = NUMBER_OF_TURNS * POINTS_PER_TURN
        z_step = Z_PER_TURN / POINTS_PER_TURN
        angle_step = 2.0 * math.pi / POINTS_PER_TURN

        safe_movel(posx(cx + CYLINDER_RADIUS, cy, z, rx, ry, rz), VEL, ACC)

        angle = 0.0
        for i in range(total):
            if stop_event.is_set():
                break
            angle += angle_step
            z += z_step
            x = cx + CYLINDER_RADIUS * math.cos(angle)
            y = cy + CYLINDER_RADIUS * math.sin(angle)
            current_progress.value = (i + 1) / total * 100
            if not safe_movel(posx(x, y, z, rx, ry, rz), DRAW_VEL, ACC):
                break

        if gripper:
            gripper.move(633)
            time.sleep(0.5)

    except Exception as e:
        print(f"[ROBOT] 오류: {e}")
    finally:
        movej(posj(*JOINT_HOME), VEL, ACC)
        if gripper:
            gripper.terminate()
        rclpy.shutdown()
        print("[ROBOT] 종료")


def robot_process_simulate(stop_event, camera_ready, hand_detected, hand_distance,
                           robot_paused, current_progress, hand_in_workspace, is_slowdown):
    """로봇 시뮬레이션"""
    camera_ready.wait()
    print("[ROBOT-SIM] 시작")

    def check_safety():
        return hand_detected.value == 0 or hand_distance.value >= SAFETY_DISTANCE_M

    def wait_clear():
        while not stop_event.is_set():
            if hand_detected.value == 0 or hand_distance.value >= SAFETY_CLEAR_DISTANCE_M:
                return True
            time.sleep(CHECK_INTERVAL)
        return False

    def safe_move(dur):
        while not stop_event.is_set():
            if check_safety():
                robot_paused.value = 0
                factor = SLOWDOWN_FACTOR if (SLOWDOWN_ENABLED and is_slowdown.value == 1) else 1.0
                time.sleep(dur / factor)
                return True
            else:
                robot_paused.value = 1
                if not wait_clear():
                    return False
        return False

    try:
        safe_move(0.5)
        safe_move(0.5)
        time.sleep(0.3)

        total = NUMBER_OF_TURNS * POINTS_PER_TURN
        for i in range(total):
            if stop_event.is_set():
                break
            current_progress.value = (i + 1) / total * 100
            if not safe_move(0.02):
                break

        time.sleep(0.3)
        print("[ROBOT-SIM] ✅ 완료")
    except Exception as e:
        print(f"[ROBOT-SIM] 오류: {e}")
    finally:
        print("[ROBOT-SIM] 종료")


# ============================================
# 모니터링 프로세스 (Socket 서버 통합)
# ============================================
def monitor_process(
    stop_event: Event,
    hand_detected: Value,
    hand_distance: Value,
    robot_paused: Value,
    current_progress: Value,
    hand_in_workspace: Value,
    is_slowdown: Value,
    calib_fx: Value,
    calib_fy: Value,
    calib_cx: Value,
    calib_cy: Value,
    calib_ok: Value,
    aruco_count: Value,
    workspace_defined: Value
):
    """모니터링 + Socket 서버"""
    print("\n" + "=" * 55)
    print("  통합 로봇 제어 시스템 - 상태 모니터")
    print("=" * 55)

    # Socket 서버 시작
    server = None
    if SOCKET_ENABLED:
        server = SafetyMonitorServer(socket_port=SOCKET_PORT, web_port=WEB_PORT)
        
        def emergency_stop():
            print("\n[SOCKET] ⚠️ 원격 비상 정지!")
            stop_event.set()
        
        server.on_emergency_stop = emergency_stop
        server.start()

    last_status = None

    try:
        while not stop_event.is_set():
            hand_det = hand_detected.value == 1
            dist = hand_distance.value
            paused = robot_paused.value == 1
            progress = current_progress.value
            in_ws = hand_in_workspace.value == 1
            slowdown = is_slowdown.value == 1

            # 상태 결정
            if hand_det and dist < 100:
                dist_cm = dist * 100
                if dist < SAFETY_DISTANCE_M:
                    status = "DANGER"
                    status_str = f"🔴 DANGER | {dist_cm:.1f}cm"
                elif dist < SAFETY_CLEAR_DISTANCE_M:
                    status = "CAUTION"
                    status_str = f"🟡 CAUTION | {dist_cm:.1f}cm"
                else:
                    status = "SAFE"
                    status_str = f"🟢 SAFE | {dist_cm:.1f}cm"
                
                if slowdown:
                    status_str += f" | 🐢 감속"
            else:
                status = "SAFE"
                status_str = "🟢 SAFE | 미감지"

            robot_str = "⏸️ PAUSE" if paused else ("🐢 SLOW" if slowdown else "▶️ RUN")
            current_str = f"{status_str} | {robot_str}"

            if current_str != last_status:
                print(f"[{time.strftime('%H:%M:%S')}] {current_str} | {progress:.1f}%")
                last_status = current_str

            # Socket 브로드캐스트
            if server:
                server.broadcast_status({
                    'status': status,
                    'hand_detected': hand_det,
                    'hand_distance_cm': round(dist * 100, 1) if dist < 100 else None,
                    'robot_paused': paused,
                    'progress': round(progress, 1),
                    'in_workspace': in_ws,
                    'is_slowdown': slowdown,
                    'calibration': {
                        'is_calibrated': calib_ok.value == 1,
                        'resolution': '1280x720',
                        'fx': round(calib_fx.value, 1) if calib_fx.value > 0 else None,
                        'fy': round(calib_fy.value, 1) if calib_fy.value > 0 else None,
                    },
                    'aruco_marker_id': ARUCO_MARKER_ID,
                    'aruco_detected': aruco_count.value,
                    'workspace_defined': workspace_defined.value == 1
                })

            time.sleep(0.1)

    finally:
        if server:
            server.stop()
        print("\n[MONITOR] 종료")


# ============================================
# 메인 함수
# ============================================
def main():
    parser = argparse.ArgumentParser(description="Doosan 로봇 통합 제어 - 완전판")
    parser.add_argument('--simulate', '-s', action='store_true', help='시뮬레이션 모드')
    parser.add_argument('--check-deps', action='store_true', help='의존성 체크')
    parser.add_argument('--no-socket', action='store_true', help='Socket 비활성화')
    parser.add_argument('--no-aruco', action='store_true', help='ArUco 비활성화')
    parser.add_argument('--no-calibration', action='store_true', help='Calibration 비활성화')
    parser.add_argument('--no-slowdown', action='store_true', help='감속 비활성화')
    parser.add_argument('--marker-id', type=int, default=0, help='ArUco 마커 ID')
    parser.add_argument('--web-port', type=int, default=8080, help='웹 모니터링 포트')
    args = parser.parse_args()

    global SOCKET_ENABLED, ARUCO_ENABLED, CALIBRATION_ENABLED, SLOWDOWN_ENABLED
    global ARUCO_MARKER_ID, WEB_PORT
    
    if args.no_socket:
        SOCKET_ENABLED = False
    if args.no_aruco:
        ARUCO_ENABLED = False
    if args.no_calibration:
        CALIBRATION_ENABLED = False
    if args.no_slowdown:
        SLOWDOWN_ENABLED = False
    ARUCO_MARKER_ID = args.marker_id
    WEB_PORT = args.web_port

    print("\n" + "=" * 60)
    print("  🤖 Doosan 로봇 통합 제어 시스템 - 완전판")
    print("=" * 60)
    print("  [3일차] Socket 통신 + 웹 모니터링:", "✅" if SOCKET_ENABLED else "❌")
    print("  [7일차] Camera Calibration:", "✅" if CALIBRATION_ENABLED else "❌")
    print("  [7일차] ArUco 작업 영역:", "✅" if ARUCO_ENABLED else "❌")
    print("  감속 기능:", "✅" if SLOWDOWN_ENABLED else "❌")
    if args.simulate:
        print("  [모드] 시뮬레이션")
    print("=" * 60 + "\n")

    if args.check_deps or not args.simulate:
        print("[MAIN] 의존성 체크...")
        vision_missing = check_vision_dependencies()
        robot_missing = check_robot_dependencies()

        if vision_missing:
            print(f"[MAIN] 비전 누락: {', '.join(vision_missing)}")
        if robot_missing:
            print(f"[MAIN] 로봇 누락: {', '.join(robot_missing)}")

        if args.check_deps:
            return

        if (vision_missing or robot_missing) and not args.simulate:
            print("\n[MAIN] 시뮬레이션 모드: python main_full.py --simulate")
            return

    # 공유 변수
    stop_event = Event()
    camera_ready = Event()
    hand_detected = Value('i', 0)
    hand_distance = Value('d', 999.0)
    robot_paused = Value('i', 0)
    current_progress = Value('d', 0.0)
    pen_x = Value('d', 0.0)
    pen_y = Value('d', 0.0)
    pen_z = Value('d', 0.0)
    pen_pixel_x = Value('i', 640)
    pen_pixel_y = Value('i', 400)
    pen_detected = Value('i', 0)
    hand_in_workspace = Value('i', 0)
    is_slowdown = Value('i', 0)
    
    # 캘리브레이션 정보
    calib_fx = Value('d', 0.0)
    calib_fy = Value('d', 0.0)
    calib_cx = Value('d', 0.0)
    calib_cy = Value('d', 0.0)
    calib_ok = Value('i', 0)
    
    # ArUco 정보
    aruco_count = Value('i', 0)
    workspace_defined = Value('i', 0)

    # 프로세스
    vision_proc = Process(
        target=vision_process,
        args=(stop_event, camera_ready, hand_detected, hand_distance,
              pen_x, pen_y, pen_z, pen_pixel_x, pen_pixel_y, pen_detected,
              hand_in_workspace, is_slowdown,
              calib_fx, calib_fy, calib_cx, calib_cy, calib_ok,
              aruco_count, workspace_defined, args.simulate),
        name="Vision"
    )

    robot_proc = Process(
        target=robot_process,
        args=(stop_event, camera_ready, hand_detected, hand_distance,
              robot_paused, current_progress, hand_in_workspace, is_slowdown, args.simulate),
        name="Robot"
    )

    monitor_proc = Process(
        target=monitor_process,
        args=(stop_event, hand_detected, hand_distance, robot_paused,
              current_progress, hand_in_workspace, is_slowdown,
              calib_fx, calib_fy, calib_cx, calib_cy, calib_ok,
              aruco_count, workspace_defined),
        name="Monitor"
    )

    def signal_handler(sig, frame):
        print("\n\n[MAIN] 종료...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        print("[MAIN] 비전 시작...")
        vision_proc.start()
        
        print("[MAIN] 카메라 대기...")
        camera_ready.wait(timeout=30)
        
        if camera_ready.is_set():
            print("[MAIN] ✅ 카메라 OK!")

        print("[MAIN] 로봇 시작...")
        robot_proc.start()

        print("[MAIN] 모니터 시작...")
        monitor_proc.start()

        print("[MAIN] 모든 프로세스 실행 중 (Ctrl+C 종료)\n")

        while not stop_event.is_set():
            if not robot_proc.is_alive():
                print("\n[MAIN] 로봇 완료")
                stop_event.set()
                break
            time.sleep(0.5)

    except Exception as e:
        print(f"[MAIN] 오류: {e}")
        stop_event.set()

    finally:
        print("\n[MAIN] 종료 중...")
        stop_event.set()

        for proc in [vision_proc, robot_proc, monitor_proc]:
            if proc.is_alive():
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.terminate()

        print("[MAIN] ✅ 완료")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()