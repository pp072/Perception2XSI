#pragma once
#include <stdio.h>
#include "DeviceInfo.h"
//#include "MyMath.h"
#include <xsisdk.h>
#include <xsi_application.h>

#include <sstream>
#include <string>
#include <fstream>

static _DeviceInfo *s_di = 0x0;
static int devider = 1;
TCHAR* GetThisPath(TCHAR* dest, size_t destSize)
{


	DWORD length = GetModuleFileName( NULL, dest, destSize );
	PathRemoveFileSpec(dest);
	return dest;
}

_DeviceInfo::_DeviceInfo()
	: m_strIPAddress(_T("192.168.1.124"))
	, m_strTCP_BHVPort(_T("7001"))
	, m_strTCP_CALCPort(_T("7009"))
{

	isRecord = false;
	isWrited = false;


	char filename[] = "settingsPN.txt";
	
	TCHAR dest[128];
	GetThisPath(dest, 128);
	std::ifstream ifs(filename);
	string line;
	vector<string> list;
	XSI::Application().LogMessage(dest);
	if (ifs) 
	{
		XSI::Application().LogMessage(L"Found config file...");
		int i = 0;
		while(!ifs.eof())
		{
			ifs >> line;
			list.push_back(line);
			if(i == 0)
				m_strIPAddress = line.c_str();
			if(i == 1)
				m_strTCP_BHVPort = line.c_str();
			if(i == 2)
				m_strTCP_CALCPort = line.c_str();
			i++;
			XSI::Application().LogMessage(line.c_str());
		}
		
	}else
		XSI::Application().LogMessage(L"Not found config file...");
	ifs.close();

	BRRegisterFrameDataCallback(this, bvhFrameDataReceived);
	BRRegisterCalculationDataCallback(this, CalcFrameDataReceive);
	
	s_di = this;
	isFirstTime = true;
}

_DeviceInfo::~_DeviceInfo()
{

}

void _DeviceInfo::connect()
{
	
	//HRESULT l_hr;

	//l_hr = l_pApp.CoCreateInstance(L"XSI.Application");

	if (sockTCPRef)
	{
		// close socket
		disconnect();

		// change the title of button
		
	}
	else
	{
		USES_CONVERSION;

		// connect to remote server
		sockTCPRef = BRConnectTo(CT2A(m_strIPAddress), atoi(m_strTCP_BHVPort));
		sockTCPRefCalc =  BRConnectTo(CT2A(m_strIPAddress), atoi(m_strTCP_CALCPort));
		// if success, change the title of button
		if(sockTCPRef)
			XSI::Application().LogMessage(L"BHV connection ... OK");
		else
			XSI::Application().LogMessage(L"BHV connection ... FALSE");
		if(sockTCPRefCalc)
			XSI::Application().LogMessage(L"CALC connection ... OK");
		else
			XSI::Application().LogMessage(L"CALC connection ... FALSE");
	}

}



void __stdcall _DeviceInfo::bvhFrameDataReceived( void* customedObj, SOCKET_REF sender, BvhDataHeader* header, float* data )
{
		s_di->showBvhBoneInfo(sender, header, data);
}

void __stdcall _DeviceInfo::CalcFrameDataReceive( void* customedObj, SOCKET_REF sender, CalcDataHeader* header, float* data )
{
	s_di->showCalcBoneInfo( sender, header, data );    
}


void _DeviceInfo::showBvhBoneInfo( SOCKET_REF sender, BvhDataHeader* header, float* data )
{
	for (int i = 0; i < NUMBONES ; i++)
	{

	
		int dataIndex = 0;
		int curSel = i;   // Gets the currently selected option in the drop down box
		if ( curSel == CB_ERR ) return;

	
		if (header->WithDisp)
		{
			dataIndex = curSel * 6;
			if (header->WithReference)
			{
				dataIndex += 6;
			}

			if(i == 0)
			{
				_sensors[i].dispX = data[dataIndex + 0];
				_sensors[i].dispY = data[dataIndex + 1];
				_sensors[i].dispZ = data[dataIndex + 2];
			}

			 _sensors[i].angX = data[dataIndex + 4];
			 _sensors[i].angY = data[dataIndex + 3];
			 _sensors[i].angZ = data[dataIndex + 5];

		// Set the X an Y channels
		
		}
		else
		{
			if (curSel == 0)
			{
				dataIndex = 0;
				if (header->WithReference)
				{
					dataIndex += 6;
				}

				if(i == 0)
				{
				// show hip's displacement
					_sensors[i].dispX  = data[dataIndex + 0];
					_sensors[i].dispY  = data[dataIndex + 1];
					_sensors[i].dispZ  = data[dataIndex + 2];
				}

			

				// show hip's angle
				_sensors[i].angX = data[dataIndex + 4];
				_sensors[i].angY = data[dataIndex + 3];
				_sensors[i].angZ = data[dataIndex + 5];


			}else
			{
				dataIndex = 3 + curSel * 3;
				if (header->WithReference)
				{
					dataIndex += 6;
				}

				_sensors[i].dispX  = 0;
				_sensors[i].dispY  = 0;
				_sensors[i].dispZ  = 0;
				// show angle
				_sensors[i].angX = data[dataIndex + 1];
				_sensors[i].angY = data[dataIndex + 0];
				_sensors[i].angZ = data[dataIndex + 2];
			}
		}
	}
}

void _DeviceInfo::showCalcBoneInfo( SOCKET_REF sender, CalcDataHeader* header, float* data )
{
	USES_CONVERSION;

	// show frame index
	char strFrameIndex[60];
	_itoa_s( header->FrameIndex, strFrameIndex, 10 );		//int transform into string by decimalism
	//GetDlgItem( IDC_STATIC_FRAME_INDEX )->SetWindowText( A2W( strFrameIndex ) );

	int dataIndex = 0;
	int curSel = 0;
	//int curSel = m_wndComBoxBone2.GetCurSel(); //Gets the currently selected option in the drop down box
	//if ( curSel == CB_ERR ) return;

	//if ( curSel > CalcBoneCount ) return;

	

	//CString tmpData;
	//tmpData.Format( L"%0.3f", data[dataIndex + 0] );
	for (int i = 0; i <  21; i++)
	{
		
		if (i == 0)
			write_data(curSel, i, dataIndex, data , HIPS);
		if (i == 1)
			write_data(curSel, i, dataIndex, data , RIGHT_UP_LEG);
		if (i == 2)
			write_data(curSel, i, dataIndex, data , RIGHT_LEG);

		if (i == 3)
			write_data(curSel, i, dataIndex, data , RIGHT_FOOT);

		if (i == 4)
			write_data(curSel, i, dataIndex, data , LEFT_UP_LEG);
		if (i == 5)
			write_data(curSel, i, dataIndex, data , LEFT_LEG);

		if (i == 6)
			write_data(curSel, i, dataIndex, data , LEFT_FOOT);



		if (i == 7)
			write_data(curSel, i, dataIndex, data , RIGHT_SHOULDER);
		if (i == 8)
			write_data(curSel, i, dataIndex, data , RIGHT_ARM);
		if (i == 9)
			write_data(curSel, i, dataIndex, data , RIGHT_FORE_ARM);
		if (i == 10)
			write_data(curSel, i, dataIndex, data , RIGHT_HAND);
		if (i == 11)
			write_data(curSel, i, dataIndex, data , LEFT_SHOULDER);
		if (i == 12)
			write_data(curSel, i, dataIndex, data , LEFT_ARM);
		if (i == 13)
			write_data(curSel, i, dataIndex, data , LEFT_FORE_ARM);
		if (i == 14)
			write_data(curSel, i, dataIndex, data , LEFT_HAND);
		if (i == 15)
			write_data(curSel, i, dataIndex, data , HEAD);
		if (i == 16)
			write_data(curSel, i, dataIndex, data , NECK);
		if (i == 17)
			write_data(curSel, i, dataIndex, data , SPINE_3);
		if (i == 18)
			write_data(curSel, i, dataIndex, data , SPINE_2);
		if (i == 19)
			write_data(curSel, i, dataIndex, data , SPINE_1);
		if (i == 20)
			write_data(curSel, i, dataIndex, data , SPINE);	
	}
}

void _DeviceInfo::write_data( int &curSel, int i, int &dataIndex, float* data, int cur_bone )
{
	curSel = i;
	dataIndex = 16 * curSel;
	_sensorsCal[cur_bone].m_calcPx = data[dataIndex + 0];
	_sensorsCal[cur_bone].m_calcPy = data[dataIndex + 1];
	_sensorsCal[cur_bone].m_calcPz = data[dataIndex + 2];
	_sensorsCal[cur_bone].m_calcVx = data[dataIndex + 3];
	_sensorsCal[cur_bone].m_calcVy = data[dataIndex + 4];
	_sensorsCal[cur_bone].m_calcVz = data[dataIndex + 5];
	_sensorsCal[cur_bone].m_calcQs = data[dataIndex + 6];
	_sensorsCal[cur_bone].m_calcQx = data[dataIndex + 7];
	_sensorsCal[cur_bone].m_calcQy = data[dataIndex + 8];
	_sensorsCal[cur_bone].m_calcQz = data[dataIndex + 9];
	_sensorsCal[cur_bone].m_clacAx = data[dataIndex + 10];
	_sensorsCal[cur_bone].m_calcAy = data[dataIndex + 11];
	_sensorsCal[cur_bone].m_calcAz = data[dataIndex + 12];
	_sensorsCal[cur_bone].m_calcGx = data[dataIndex + 13];
	_sensorsCal[cur_bone].m_calcGy = data[dataIndex + 14];
	_sensorsCal[cur_bone].m_calcGz = data[dataIndex + 15];
}

void _DeviceInfo::disconnect()
{
	if (sockTCPRef)
	{
		// close socket
		BRCloseSocket(sockTCPRef);
		BRCloseSocket(sockTCPRefCalc);

		sockTCPRef = NULL;
		sockTCPRefCalc = NULL;

	}
}


