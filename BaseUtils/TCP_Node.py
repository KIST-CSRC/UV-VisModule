#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ##s
# @brief    [BaseTCPNode] basic function about TCP connection
# @author   Hyuk Jun Yoo (yoohj9475@kist.re.kr)   
# @version  1_1   
# TEST 2021-11-14

import socket
import os, sys
import json
import time


class BaseTCPNode(object):

    def __init__(self):
        self.BUFF_SIZE = 4096
    
    def checkSocketStatus(self, client_socket, res_msg, hardware_name, action_type):
        if bool(res_msg) == True:
            if type(res_msg)==dict: # 
                ourbyte=b''
                ourbyte = json.dumps(res_msg).encode("utf-8")
                self._sendTotalJSON(client_socket, ourbyte)
                time.sleep(3)
                finish_msg="finish"
                client_socket.sendall(finish_msg.encode())
            else:
                cmd_string_end = "[{}] {} action success".format(hardware_name, action_type)
                client_socket.sendall(cmd_string_end.encode())
        elif bool(res_msg) == False:
            cmd_string_end = "[{}] {} action error".format(hardware_name, action_type)
            client_socket.sendall(cmd_string_end.encode())
            raise ConnectionError("{} : Please check".format(cmd_string_end))

    def callServer(self, host, port, command_byte):
        res_msg=""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            time.sleep(2)
            s.sendall(command_byte)
            print("Send : ",command_byte)
            msg = b''
            while True:
                part = s.recv(self.BUFF_SIZE)
                msg += part
                if len(part) < self.BUFF_SIZE:
                    s.close()
                    break
            res_msg=msg.decode('UTF-8')
        return res_msg
    
    def _sendTotalJSON(self, client_socket, ourbyte):
        cnt=0
        while (cnt+1)*self.BUFF_SIZE < len(ourbyte):
            msg_temp = b""+ourbyte[cnt * self.BUFF_SIZE: (cnt + 1) * self.BUFF_SIZE]
            client_socket.sendall(msg_temp)
            cnt += 1
        msg_temp = b"" + ourbyte[cnt * self.BUFF_SIZE: len(ourbyte)]
        client_socket.sendall(msg_temp)