#pragma once
//#include "DataType.h"
#include <assert.h>            
#include <atlbase.h>            // ATL.
// #include <xsisdk.h>  // Softimage object model interfaces.
// #include <sicppsdk.h>
// #include <xsi_application.h>
 #include <xsi_fcurve.h>
// #include "xsi_model.h"
// #include <xsi_null.h>
// #include <xsi_project.h>

#include <xsi_x3dobject.h>
#include <atlstr.h>
#include <vector>
#include <xsi_sceneitem.h>
using namespace std;

#include "NeuronDataReader.h"
#define NUMBONES 59

#define HIPS 0

#define RIGHT_UP_LEG 1
#define RIGHT_LEG 2
#define RIGHT_FOOT 3
#define LEFT_UP_LEG 4
#define LEFT_LEG 5
#define LEFT_FOOT 6

#define SPINE 7
#define SPINE_1 8
#define SPINE_2 9
#define SPINE_3 10

#define NECK 11
#define HEAD 12

#define RIGHT_SHOULDER 13
#define RIGHT_ARM 14
#define RIGHT_FORE_ARM 15
#define RIGHT_HAND 16

#define LEFT_SHOULDER 36
#define LEFT_ARM 37
#define LEFT_FORE_ARM 38
#define LEFT_HAND 39

//FINGERS RIGHT
#define RIGHT_HAND_THUMB_1 17
#define RIGHT_HAND_THUMB_2 18
#define RIGHT_HAND_THUMB_3 19

#define RIGHT_IN_HAND_INDEX 20

#define RIGHT_HAND_INDEX_1 21
#define RIGHT_HAND_INDEX_2 22
#define RIGHT_HAND_INDEX_3 23

#define RIGHT_IN_HAND_MIDDLE 24

#define RIGHT_HAND_MIDDLE_1 25
#define RIGHT_HAND_MIDDLE_2 26
#define RIGHT_HAND_MIDDLE_3 27

#define RIGHT_IN_HAND_RING 28

#define RIGHT_HAND_RING_1 29
#define RIGHT_HAND_RING_2 30
#define RIGHT_HAND_RING_3 31

#define RIGHT_IN_HAND_PINKY 32

#define RIGHT_HAND_PINKY_1 33
#define RIGHT_HAND_PINKY_2 34
#define RIGHT_HAND_PINKY_3 35

//FINGERS LEFT
#define LEFT_HAND_THUMB_1 40
#define LEFT_HAND_THUMB_2 41
#define LEFT_HAND_THUMB_3 42

#define LEFT_IN_HAND_INDEX 43

#define LEFT_HAND_INDEX_1 44
#define LEFT_HAND_INDEX_2 45
#define LEFT_HAND_INDEX_3 46

#define LEFT_IN_HAND_MIDDLE 47

#define LEFT_HAND_MIDDLE_1 48
#define LEFT_HAND_MIDDLE_2 49
#define LEFT_HAND_MIDDLE_3 50

#define LEFT_IN_HAND_RING 51

#define LEFT_HAND_RING_1 52
#define LEFT_HAND_RING_2 53
#define LEFT_HAND_RING_3 54

#define LEFT_IN_HAND_PINKY 55

#define LEFT_HAND_PINKY_1 56
#define LEFT_HAND_PINKY_2 57
#define LEFT_HAND_PINKY_3 58
//#include "afxwin.h"

struct key
{
	float frame;
	float value;
};
struct sensor
{


	float dispX ;
	float dispY ;
	float dispZ ;

	float angX ;
	float angY ;
	float angZ ;

	XSI::FCurve fc_RX;
	XSI::FCurve fc_RY;
	XSI::FCurve fc_RZ;

	XSI::FCurve fc_TX;
	XSI::FCurve fc_TY;
	XSI::FCurve fc_TZ;

	vector<key> fc_tmp_RX;
	vector<key> fc_tmp_RY;
	vector<key> fc_tmp_RZ;

	vector<key> fc_tmp_TX;
	vector<key> fc_tmp_TY;
	vector<key> fc_tmp_TZ;
	
};
struct sensor_calc
{


	float m_calcPx ;
	float m_calcPy ;
	float m_calcPz ;
	float m_calcVx ;
	float m_calcVy ;
	float m_calcVz ;
	float m_calcQs ;
	float m_calcQx ;
	float m_calcQy ;
	float m_calcQz ;
	float m_clacAx ;
	float m_calcAy ;
	float m_calcAz ;
	float m_calcGx ;
	float m_calcGy ;
	float m_calcGz ;
};
class _DeviceInfo
{
public:
	_DeviceInfo();
	~_DeviceInfo();
	
	XSI::X3DObject RootNodePN;
	XSI::X3DObject Hips;

	sensor _sensors[NUMBONES];
	sensor_calc _sensorsCal[21];

	float _last_frame;
	bool isFirstTime;
	


	bool isRecord;
	bool isWrited;
	bool isActivated;

	ATL::CString m_strIPAddress;
	ATL::CString m_strTCP_BHVPort;
	ATL::CString m_strTCP_CALCPort;

	
	enum
	{ 
		BVHBoneCount = 17,
		CalcBoneCount = 21,
	};

	static void __stdcall bvhFrameDataReceived(void* customedObj, SOCKET_REF sender, BvhDataHeader* header, float* data);
	// receive Calc data
	static void __stdcall CalcFrameDataReceive( void* customedObj, SOCKET_REF sender, CalcDataHeader* header, float* data );

	void showBvhBoneInfo(SOCKET_REF sender, BvhDataHeader* header, float* data);
	void showCalcBoneInfo(SOCKET_REF sender, CalcDataHeader* header, float* data );

	void write_data( int &curSel, int i, int &dataIndex, float* data , int cur_bone);

	SOCKET_REF sockTCPRef;
	SOCKET_REF sockTCPRefCalc;

	

	void connect();
	void disconnect();
private:
	
};



