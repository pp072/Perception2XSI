#pragma once

#include <math.h>
#include <float.h>
////#define //stop __asm nop

// Constants

namespace Math
{
	const unsigned int MaxUInt32 = 0xFFFFFFFF;
	const int MinInt32 = 0x80000000;
	const int MaxInt32 = 0x7FFFFFFF;
	const float MaxFloat = FLT_MAX;

	const float Pi = 3.1415926f;
	const float TwoPi = Pi * 2;
	const float PiHalf = Pi / 2;

	const float Epsilon = 0.000001f;
	const float NaN = *(float *)&MaxUInt32;
};

// -------------------------------------------------------------------------------------------------
// General
// -------------------------------------------------------------------------------------------------

inline float degToRad( float f ) 
{
	return f * (Math::Pi / 180.0f);
}

inline float radToDeg( float f ) 
{
	return f * (180.0f / Math::Pi);
}
inline bool closeEnough(float f1, float f2)
{
	// Determines whether the two floating-point values f1 and f2 are
	// close enough together that they can be considered equal.

	return fabsf((f1 - f2) / ((f2 == 0.0f) ? 1.0f : f2)) < Math::Epsilon;
}
inline float clamp( float f, float min, float max )
{
	if( f < min ) f = min;
	if( f > max ) f = max;

	return f;
}

inline float minf( float a, float b )
{
	return a < b ? a : b;
}

inline float maxf( float a, float b )
{
	return a > b ? a : b;
}


// -------------------------------------------------------------------------------------------------
// Vector
// -------------------------------------------------------------------------------------------------

class Vec3f
{
public:
	float x, y, z;


	// ------------
	// Constructors
	// ------------
	Vec3f() : x( 0.0f ), y( 0.0f ), z( 0.0f ) 
	{ 
	}

	explicit Vec3f( const float x, const float y, const float z ) : x( x ), y( y ), z( z ) 
	{
	}
	void Set(float _x,float _y,float _z)
	{
		x=_x; y=_y; z=_z;
	}
	Vec3f& Normalize (void)                       /// NORMALIZE VECTOR
	{ 
		float Lenght = Length();                       //  CALCULATE LENGTH
		if (Lenght>0) 
		{ 
			x/=Lenght; y/=Lenght; z/=Lenght; 
		}           
	return *this;
	}
	
	float Length (void) const                     /// LENGTH OF VECTOR
	{ 
		return (sqrt(x*x+y*y+z*z) );
	}
	float magnitudeSq() const
	{
		return (x * x) + (y * y) + (z * z);
	}
	// -----------
	// Comparisons
	// -----------
	bool operator==( const Vec3f &v ) const
	{
		return (x > v.x - Math::Epsilon && x < v.x + Math::Epsilon && 
			y > v.y - Math::Epsilon && y < v.y + Math::Epsilon &&
			z > v.z - Math::Epsilon && z < v.z + Math::Epsilon);

	}

	bool operator!=( const Vec3f &v ) const
	{
		return (x < v.x - Math::Epsilon || x > v.x + Math::Epsilon || 
			y < v.y - Math::Epsilon || y > v.y + Math::Epsilon ||
			z < v.z - Math::Epsilon || z > v.z + Math::Epsilon);

	}

	// ---------------------
	// Artitmetic operations
	// ---------------------
	Vec3f operator-() const
	{
		return Vec3f( -x, -y, -z );
	}

	Vec3f operator+( const Vec3f &v ) const
	{
		return Vec3f( x + v.x, y + v.y, z + v.z );
	}

	Vec3f &operator+=( const Vec3f &v )
	{
		return *this = *this + v;
	}

	Vec3f operator-( const Vec3f &v ) const 
	{
		return Vec3f( x - v.x, y - v.y, z - v.z );
	}

	Vec3f &operator-=( const Vec3f &v )
	{
		return *this = *this - v;
	}

	Vec3f operator*( const float f ) const
	{
		return Vec3f( x * f, y * f, z * f );
	}

	Vec3f &operator*=( const float f )
	{
		return *this = *this * f;
	}

	Vec3f operator/( const float f ) const
	{
		return Vec3f( x / f, y / f, z / f );
	}

	Vec3f &operator/=( const float f )
	{
		return *this = *this / f;
	}
	inline float squaredLength () const
	{
		return x * x + y * y + z * z;
	}

	// -----------
	// Dot product
	// -----------
	float operator*( const Vec3f &v ) const
	{
		return ( x * v.x + y * v.y + z * v.z );
	}

	// -------------
	// Cross Product
	// -------------
	Vec3f crossProduct( const Vec3f &v ) const
	{
		return Vec3f( y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x );
	}

	// ----------------
	// Other operations
	// ----------------
	float length() const 
	{
		return sqrtf( *this * *this );
	}

	Vec3f normalized() const
	{
		float l = length();

		if( l != 0 ) return Vec3f( x / l, y / l, z / l );
		else return Vec3f( 0, 0, 0 );
	}

	/*void fromRotation( float angleX, float angleY )
	{
	x = cosf( angleX ) * sinf( angleY ); 
	y = -sinf( angleX );
	z = cosf( angleX ) * cosf( angleY );
	}*/
	inline float normalise()
	{
		float fLength = sqrt( x * x + y * y + z * z );

		// Will also work for zero-sized vectors, but will change nothing
		if ( fLength > 1e-08 )
		{
			float fInvLength = 1.0 / fLength;
			x *= fInvLength;
			y *= fInvLength;
			z *= fInvLength;
		}

		return fLength;
	}
	Vec3f toRotation() const
	{
		// Assumes that the unrotated view vector is (0, 0, -1)
		Vec3f v;

		if( y != 0 ) v.x = atan2( y, sqrtf( x*x + z*z ) );
		if( z != 0 ) v.y = atan2( -x, -z );

		return v;
	}

	Vec3f lerp( const Vec3f &v, float f ) const
	{
		return Vec3f( x + (v.x - x) * f, y + (v.y - y) * f, z + (v.z - z) * f ); 
	}
};


class Vec4f
{
public:

	float x, y, z, w;


	Vec4f() : x( 0 ), y( 0 ), z( 0 ), w( 0 )
	{
	}

	explicit Vec4f( const float x, const float y, const float z, const float w )
		: x( x ), y( y ), z( z ), w( w )
	{
	}

	explicit Vec4f( Vec3f v ) : x( v.x ), y( v.y ), z( v.z ), w( 1.0f )
	{
	}
};


// -------------------------------------------------------------------------------------------------
// Quaternion
// -------------------------------------------------------------------------------------------------

class Quaternion
{
public:	

	float x, y, z, w;

	// ------------
	// Constructors
	// ------------
	Quaternion() : x( 0.0f ), y( 0.0f ), z( 0.0f ), w( 1.0f ) 
	{ 
	}

	explicit Quaternion( const float x, const float y, const float z, const float w ) :
	x( x ), y( y ), z( z ), w( w )
	{
	}
	void FromAxes(const Vec3f &xaxis,const Vec3f& yaxis, const Vec3f& zaxis)
	{
		//Matrix4f kRot;
		float kRot[4][4];
		kRot[0][0] = 1; kRot[1][0] = 0; kRot[2][0] = 0; kRot[3][0] = 0;
		kRot[0][1] = 0; kRot[1][1] = 1; kRot[2][1] = 0; kRot[3][1] = 0;
		kRot[0][2] = 0; kRot[1][2] = 0; kRot[2][2] = 1; kRot[3][2] = 0;
		kRot[0][3] = 0; kRot[1][3] = 0; kRot[2][3] = 0; kRot[3][3] = 1;
		//Matrix4f

		kRot[0][0] = xaxis.x;
		kRot[1][0] = xaxis.y;
		kRot[2][0] = xaxis.z;

		kRot[0][1] = yaxis.x;
		kRot[1][1] = yaxis.y;
		kRot[2][1] = yaxis.z;

		kRot[0][2] = zaxis.x;
		kRot[1][2] = zaxis.y;
		kRot[2][2] = zaxis.z;

		FromRotationMatrix(kRot);
		////stop
	}
	void FromRotationMatrix(float (*kRot)[4])
	{
		////stop
		float trace = kRot[0][0] + kRot[1][1] + kRot[2][2]; // I removed + 1.0f; see discussion with Ethan
		if( trace > 0 ) {// I changed M_EPSILON to 0
			float s = 0.5f / sqrtf(trace+ 1.0f);
			w = 0.25f / s;
			x = ( kRot[2][1] - kRot[1][2] ) * s;
			y = ( kRot[0][2] - kRot[2][0] ) * s;
			z = ( kRot[1][0] - kRot[0][1] ) * s;
		} else {
			if ( kRot[0][0] > kRot[1][1] && kRot[0][0] > kRot[2][2] ) {
				float s = 2.0f * sqrtf( 1.0f + kRot[0][0] - kRot[1][1] - kRot[2][2]);
				w = (kRot[2][1] - kRot[1][2] ) / s;
				x = 0.25f * s;
				y = (kRot[0][1] + kRot[1][0] ) / s;
				z = (kRot[0][2] + kRot[2][0] ) / s;
			} else if (kRot[1][1] > kRot[2][2]) {
				float s = 2.0f * sqrtf( 1.0f + kRot[1][1] - kRot[0][0] - kRot[2][2]);
				w = (kRot[0][2] - kRot[2][0] ) / s;
				x = (kRot[0][1] + kRot[1][0] ) / s;
				y = 0.25f * s;
				z = (kRot[1][2] + kRot[2][1] ) / s;
			} else {
				float s = 2.0f * sqrtf( 1.0f + kRot[2][2] - kRot[0][0] - kRot[1][1] );
				w = (kRot[1][0] - kRot[0][1] ) / s;
				x = (kRot[0][2] + kRot[2][0] ) / s;
				y = (kRot[1][2] + kRot[2][1] ) / s;
				z = 0.25f * s;
			}
		}
	}
	void FromRotationMatrixView(float (*kRot)[4])
	{
		////stop

		float s = 0.0f;
		float q[4] = {0.0f};
		float trace = kRot[0][0] + kRot[1][1] + kRot[2][2];

		if (trace > 0.0f)
		{
			s = sqrtf(trace + 1.0f);
			q[3] = s * 0.5f;
			s = 0.5f / s;
			q[0] = (kRot[1][2] - kRot[2][1]) * s;
			q[1] = (kRot[2][0] - kRot[0][2]) * s;
			q[2] = (kRot[0][1] - kRot[1][0]) * s;
		}
		else
		{
			int nxt[3] = {1, 2, 0};
			int i = 0, j = 0, k = 0;

			if (kRot[1][1] > kRot[0][0])
				i = 1;

			if (kRot[2][2] > kRot[i][i])
				i = 2;

			j = nxt[i];
			k = nxt[j];
			s = sqrtf((kRot[i][i] - (kRot[j][j] + kRot[k][k])) + 1.0f);

			q[i] = s * 0.5f;
			s = 0.5f / s;
			q[3] = (kRot[j][k] - kRot[k][j]) * s;
			q[j] = (kRot[i][j] + kRot[j][i]) * s;
			q[k] = (kRot[i][k] + kRot[k][i]) * s;
		}

		x = q[0], y = q[1], z = q[2], w = q[3];
	}
	void ToAxes (Vec3f* akAxis) const
	{
		float kRot[4][4];
		kRot[0][0] = 1; kRot[1][0] = 0; kRot[2][0] = 0; kRot[3][0] = 0;
		kRot[0][1] = 0; kRot[1][1] = 1; kRot[2][1] = 0; kRot[3][1] = 0;
		kRot[0][2] = 0; kRot[1][2] = 0; kRot[2][2] = 1; kRot[3][2] = 0;
		kRot[0][3] = 0; kRot[1][3] = 0; kRot[2][3] = 0; kRot[3][3] = 1;

		ToRotationMatrix(kRot);

		for (size_t iCol = 0; iCol < 3; iCol++)
		{
			akAxis[iCol].x = kRot[0][iCol];
			akAxis[iCol].y = kRot[1][iCol];
			akAxis[iCol].z = kRot[2][iCol];
		}
	}
	void ToRotationMatrix (float (*kRot)[4]) const
	{
		float fTx  = 2.0*x;
		float fTy  = 2.0*y;
		float fTz  = 2.0*z;
		float fTwx = fTx*w;
		float fTwy = fTy*w;
		float fTwz = fTz*w;
		float fTxx = fTx*x;
		float fTxy = fTy*x;
		float fTxz = fTz*x;
		float fTyy = fTy*y;
		float fTyz = fTz*y;
		float fTzz = fTz*z;

		kRot[0][0] = 1.0-(fTyy+fTzz);
		kRot[0][1] = fTxy-fTwz;
		kRot[0][2] = fTxz+fTwy;
		kRot[1][0] = fTxy+fTwz;
		kRot[1][1] = 1.0-(fTxx+fTzz);
		kRot[1][2] = fTyz-fTwx;
		kRot[2][0] = fTxz-fTwy;
		kRot[2][1] = fTyz+fTwx;
		kRot[2][2] = 1.0-(fTxx+fTyy);
	}
	void ToRotationMatrixView (float (*kRot)[4]) const
	{
		float x2 = x + x; 
		float y2 = y + y; 
		float z2 = z + z;
		float xx = x * x2;
		float xy = x * y2;
		float xz = x * z2;
		float yy = y * y2;
		float yz = y * z2;
		float zz = z * z2;
		float wx = w * x2;
		float wy = w * y2;
		float wz = w * z2;

		//Matrix4 m;

		kRot[0][0] = 1.0f - (yy + zz);
		kRot[0][1] = xy + wz;
		kRot[0][2] = xz - wy;
		kRot[0][3] = 0.0f;

		kRot[1][0] = xy - wz;
		kRot[1][1] = 1.0f - (xx + zz);
		kRot[1][2] = yz + wx;
		kRot[1][3] = 0.0f;

		kRot[2][0] = xz + wy;
		kRot[2][1] = yz - wx;
		kRot[2][2] = 1.0f - (xx + yy);
		kRot[2][3] = 0.0f;

		kRot[3][0] = 0.0f;
		kRot[3][1] = 0.0f;
		kRot[3][2] = 0.0f;
		kRot[3][3] = 1.0f;
	}
// 	inline void to_axis_angle(Vec3f& axis, float& angle)const {
// 		float vl = (float)sqrt( x*x + y*y + z*z );
// 		if( vl > TINY )
// 		{
// 			float ivl = 1.0f/vl;
// 			axis.set( x*ivl, y*ivl, z*ivl );
// 			if( w < 0 )
// 				angle = 2.0f*(float)atan2(-vl, -w); //-PI,0 
// 			else
// 				angle = 2.0f*(float)atan2( vl,  w); //0,PI 
// 		}else{
// 			axis = vector3(0,0,0);
// 			angle = 0;
// 		}
// 	};

	void FromAngleAxis ( float angle,const Vec3f& v)
	{
		// assert:  axis[] is unit length
		//
		// The quaternion representing the rotation is
		//   q = cos(A/2)+sin(A/2)*(x*i+y*j+z*k)

		float sinAngle;
		angle *= 0.5f;
		Vec3f vn(v);
		vn.normalise();

		sinAngle = asinf(angle);

		x = (vn.x * sinAngle);
		y = (vn.y * sinAngle);
		z = (vn.z * sinAngle);
		w = acosf(angle);

	}
	void fromAxisAngle(const Vec3f &axis, float degrees)
	{
		float halfTheta = degToRad(degrees) * 0.5f;
		float s = sinf(halfTheta);
		w = cosf(halfTheta), x = axis.x * s, y = axis.y * s, z = axis.z * s;
	}
	float normalise(void)
	{
		float len = Norm();
		float factor = 1.0f / sqrt(len);
		*this = *this * factor;
		return len;
	}
	float Quaternion::Norm () const
	{
		return w*w+x*x+y*y+z*z;
	}
	Quaternion operator* (float fScalar) const
	{
		return Quaternion(fScalar*w,fScalar*x,fScalar*y,fScalar*z);
	}
	Quaternion( const float eulerX, const float eulerY, const float eulerZ )
	{
		Quaternion roll( sinf( eulerX / 2 ), 0, 0, cosf( eulerX / 2 ) );
		Quaternion pitch( 0, sinf( eulerY / 2 ), 0, cosf( eulerY / 2 ) );
		Quaternion yaw( 0, 0, sinf( eulerZ / 2 ), cosf( eulerZ / 2 ) );

		// Order: y * x * z
		*this = pitch * roll * yaw;
	}
	//Quaternion operator* (Real fScalar) const;

	// ---------------------
	// Artitmetic operations
	// ---------------------
	Quaternion operator+ (const Quaternion& rkQ) const
	{
		return Quaternion(w+rkQ.w,x+rkQ.x,y+rkQ.y,z+rkQ.z);
	}
	Quaternion operator*( const Quaternion &q ) const
	{
		return Quaternion(
			y * q.z - z * q.y + q.x * w + x * q.w,
			z * q.x - x * q.z + q.y * w + y * q.w,
			x * q.y - y * q.x + q.z * w + z * q.w,
			w * q.w - (x * q.x + y * q.y + z * q.z) );
	}
	Quaternion mult( const Quaternion &q ) const
	{
		return Quaternion(
			(w * q.x) + (x * q.w) - (y * q.z) + (z * q.y),
			(w * q.y) + (x * q.z) + (y * q.w) - (z * q.x),
			(w * q.z) - (x * q.y) + (y * q.x) + (z * q.w),
			(w * q.w) - (x * q.x) - (y * q.y) - (z * q.z));
	}


	Quaternion &operator*=( const Quaternion &q )
	{
		return *this = *this * q;
	}
	Vec3f operator* (const Vec3f& rkVector) const
	{
		Vec3f uv, uuv;
		Vec3f qvec(x, y, z);
		uv = qvec.crossProduct(rkVector);
		uuv = qvec.crossProduct(uv);
		uv *= (2.0f * w);
		uuv *= 2.0f;

		return rkVector + uv + uuv;
	}
	// ----------------
	// Other operations
	// ----------------

	Quaternion nlerp( const Quaternion &q, const float t ) const
	{
		// Normalized linear quaternion interpolation
		// Note: NLERP is faster than SLERP and commutative but does not yield constant velocity

		Quaternion qt;
		float cosTheta = x * q.x + y * q.y + z * q.z + w * q.w;

		// Use the shortest path and interpolate linearly
		if( cosTheta < 0 )
			qt = Quaternion( x + (-q.x - x) * t, y + (-q.y - y) * t,
			z + (-q.z - z) * t, w + (-q.w - w) * t );
		else
			qt = Quaternion( x + (q.x - x) * t, y + (q.y - y) * t,
			z + (q.z - z) * t, w + (q.w - w) * t );

		// Return normalized quaternion
		float invLen = 1.0f / sqrtf( qt.x * qt.x + qt.y * qt.y + qt.z * qt.z + qt.w * qt.w );
		return Quaternion( qt.x * invLen, qt.y * invLen, qt.z * invLen, qt.w * invLen );
	}

	Quaternion slerp( const Quaternion &q, const float t ) const
	{
		float		omega, cosom, sinom, scale0, scale1;
		Quaternion	q1 = q, res;

		// Calc cosine
		cosom = x * q.x + y * q.y + z * q.z + w * q.w;

		// Adjust signs (if necessary)
		if( cosom < 0 ) {
			cosom = -cosom; 
			q1.x = -q.x;
			q1.y = -q.y;
			q1.z = -q.z;
			q1.w = -q.w;
		} 

		// Calculate coefficients
		if( (1 - cosom) > Math::Epsilon ) {
			// Standard case (Slerp)
			omega = acosf( cosom );
			sinom = sinf( omega );
			scale0 = sinf( (1 - t) * omega ) / sinom;
			scale1 = sinf( t * omega ) / sinom;
		} 
		else {        
			// Quaternions very close, so do linear interpolation
			scale0 = 1 - t;
			scale1 = t;
		}

		// Calculate final values
		res.x = x * scale0 + q1.x * scale1;
		res.y = y * scale0 + q1.y * scale1;
		res.z = z * scale0 + q1.z * scale1;
		res.w = w * scale0 + q1.w * scale1;

		return res;
	}

	Quaternion inverted() const
	{
		float len = x * x + y * y + z * z + w * w;
		if( len > 0 )
		{
			float invLen = 1.0f / len;
			return Quaternion( -x * invLen, -y * invLen, -z * invLen, w * invLen );
		}
		else return Quaternion();
	}
	void ToMatrix(float matrix[16]) {
		matrix[0]  = (1.0f - (2.0f * ((y * y) + (z * z))));
		matrix[1]  =         (2.0f * ((x * y) + (z * w)));
		matrix[2]  =         (2.0f * ((x * z) - (y * w)));
		matrix[3]  = 0.0f;
		matrix[4]  =         (2.0f * ((x * y) - (z * w)));
		matrix[5]  = (1.0f - (2.0f * ((x * x) + (z * z))));
		matrix[6]  =         (2.0f * ((y * z) + (x * w)));
		matrix[7]  = 0.0f;
		matrix[8]  =         (2.0f * ((x * z) + (y * w)));
		matrix[9]  =         (2.0f * ((y * z) - (x * w)));
		matrix[10] = (1.0f - (2.0f * ((x * x) + (y * y))));
		matrix[11] = 0.0f;
		matrix[12] = 0.0f;
		matrix[13] = 0.0f;
		matrix[14] = 0.0f;
		matrix[15] = 1.0f;
	}
};

inline void  QuaternionFromTwoDirs2( Quaternion& quat, const Vec3f& From, const Vec3f& To)
{
	Vec3f fromtmp = From;
	//Vec3f c = From.crossProduct(To);
	float     CosA  = fromtmp * To ;

	if( CosA < - 0.99999f ){  // angle close to PI    ( can replaced by Bisect.lensquared() < 0.000001f ) ;
		Vec3f CrossVec( 0, fromtmp.x, -fromtmp.y ); // cross with (1, 0, 0)
		if( ( fromtmp.z*fromtmp.z ) > ( fromtmp.y*fromtmp.y ) )
		{
			// if (0, 1, 0) Cross > (1, 0, 0) Cross
			CrossVec =Vec3f( -fromtmp.z, 0, fromtmp.x ); // cross with (0 ,1, 0)
		}
		CrossVec.normalise();
		quat=Quaternion( CrossVec.x, CrossVec.y, CrossVec.z, 0.0f );
	}else{
		Vec3f Bisect( fromtmp + To );
		Bisect.normalise();
		
		Vec3f BCross = fromtmp.crossProduct(Bisect);
		quat=Quaternion( BCross.x, BCross.y, BCross.z, fromtmp* Bisect) ;
	}



	
}

// -------------------------------------------------------------------------------------------------
// Matrix
// -------------------------------------------------------------------------------------------------

class Matrix4f
{
private:

	Matrix4f( bool )
	{
		// Don't initialize the matrix
	}

public:

	union
	{
		float c[4][4];	// Column major order for OpenGL: c[column][row]
		float x[16];
	};

	// --------------
	// Static methods
	// --------------
	static Matrix4f TransMat( float x, float y, float z )
	{
		Matrix4f m;

		m.c[3][0] = x;
		m.c[3][1] = y;
		m.c[3][2] = z;

		return m;
	}

	static Matrix4f ScaleMat( float x, float y, float z )
	{
		Matrix4f m;

		m.c[0][0] = x;
		m.c[1][1] = y;
		m.c[2][2] = z;

		return m;
	}

	static Matrix4f RotMat( float x, float y, float z )
	{
		// Rotation order: YXZ [* Vector]
		return Matrix4f( Quaternion( x, y, z ) );
	}

	static Matrix4f RotMat( Vec3f axis, float angle )
	{
		axis = axis * sinf( angle / 2 );
		return Matrix4f( Quaternion( axis.x, axis.y, axis.z, cosf( angle / 2 ) ) );
	}
	static Matrix4f PerspectiveMat( float l, float r, float b, float t, float n, float f )
	{
		Matrix4f m;

		m.x[0] = 2 * n / (r - l);
		m.x[5] = 2 * n / (t - b);
		m.x[8] = (r + l) / (r - l);
		m.x[9] = (t + b) / (t - b);
		m.x[10] = -(f + n) / (f - n);
		m.x[11] = -1;
		m.x[14] = -2 * f * n / (f - n);
		m.x[15] = 0;

		return m;
	}
	void rotate(const Vec3f &axis, float degrees )
	{
		degrees = degToRad(degrees);

		float x = axis.x;
		float y = axis.y;
		float z = axis.z;
		float co = cosf(degrees);
		float so = sinf(degrees);

		c[0][0] = (x * x) * (1.0f - co) + co;
		c[0][1] = (x * y) * (1.0f - co) + (z * so);
		c[0][2] = (x * z) * (1.0f - co) - (y * so);
		c[0][3] = 0.0f;

		c[1][0] = (y * x) * (1.0f - co) - (z * so);
		c[1][1] = (y * y) * (1.0f - co) + co;
		c[1][2] = (y * z) * (1.0f - co) + (x * so);
		c[1][3] = 0.0f;

		c[2][0] = (z * x) * (1.0f - co) + (y * so);
		c[2][1] = (z * y) * (1.0f - co) - (x * so);
		c[2][2] = (z * z) * (1.0f - co) + co;
		c[2][3] = 0.0f;

		c[3][0] = 0.0f;
		c[3][1] = 0.0f;
		c[3][2] = 0.0f;
		c[3][3] = 1.0f;
	}
	void RotateX(float deg)
	{

		float  sr = sin( degToRad(deg) );
		float  cr = cos( degToRad(deg) );
		c[1][1] =  cr;
		c[2][1] = -sr;
		c[1][2] =  sr;
		c[2][2] =  cr;
	}
	void RotateY(float deg)
	{

		float  sr = sin( degToRad(deg) );
		float  cr = cos( degToRad(deg) );
		c[0][0] =  cr;
		c[2][0] =  sr;
		c[0][2] = -sr;
		c[2][2] =  cr;
	}
	void RotateZ(float deg)
	{

		float  sr = sin( degToRad(deg) );
		float  cr = cos( degToRad(deg) );
		c[0][0] =  cr;
		c[1][0] = -sr;
		c[0][1] =  sr;
		c[1][1] =  cr;
	}
	void fromHeadPitchRoll(float headDegrees, float pitchDegrees, float rollDegrees)
	{

		headDegrees = degToRad(headDegrees);
		pitchDegrees = degToRad(pitchDegrees);
		rollDegrees = degToRad(rollDegrees);

		float cosH = cosf(headDegrees);
		float cosP = cosf(pitchDegrees);
		float cosR = cosf(rollDegrees);
		float sinH = sinf(headDegrees);
		float sinP = sinf(pitchDegrees);
		float sinR = sinf(rollDegrees);

		c[0][0] = cosR * cosH - sinR * sinP * sinH;
		c[0][1] = sinR * cosH + cosR * sinP * sinH;
		c[0][2] = -cosP * sinH;

		c[1][0] = -sinR * cosP;
		c[1][1] = cosR * cosP;
		c[1][2] = sinP;

		c[2][0] = cosR * sinH + sinR * sinP * cosH;
		c[2][1] = sinR * sinH - cosR * sinP * cosH;
		c[2][2] = cosP * cosH;
	}
	// ------------
	// Constructors
	// ------------
	Matrix4f()
	{
		c[0][0] = 1; c[1][0] = 0; c[2][0] = 0; c[3][0] = 0;
		c[0][1] = 0; c[1][1] = 1; c[2][1] = 0; c[3][1] = 0;
		c[0][2] = 0; c[1][2] = 0; c[2][2] = 1; c[3][2] = 0;
		c[0][3] = 0; c[1][3] = 0; c[2][3] = 0; c[3][3] = 1;
	}

	Matrix4f( const float *floatArray16 )
	{
		for( unsigned int i = 0; i < 4; ++i )
		{
			for( unsigned int j = 0; j < 4; ++j )
			{
				c[i][j] = floatArray16[i * 4 + j];
			}
		}
	}

	Matrix4f( const Quaternion &q )
	{
		float wx, wy, wz, xx, yy, yz, xy, xz, zz, x2, y2, z2;

		// Calculate coefficients
		x2 = q.x + q.x;	y2 = q.y + q.y;	z2 = q.z + q.z;
		xx = q.x * x2;	xy = q.x * y2;	xz = q.x * z2;
		yy = q.y * y2;	yz = q.y * z2;	zz = q.z * z2;
		wx = q.w * x2;	wy = q.w * y2;	wz = q.w * z2;


		c[0][0] = 1 - (yy + zz);	c[1][0] = xy - wz;	
		c[2][0] = xz + wy;			c[3][0] = 0;
		c[0][1] = xy + wz;			c[1][1] = 1 - (xx + zz);
		c[2][1] = yz - wx;			c[3][1] = 0;
		c[0][2] = xz - wy;			c[1][2] = yz + wx;
		c[2][2] = 1 - (xx + yy);	c[3][2] = 0;
		c[0][3] = 0;				c[1][3] = 0;
		c[2][3] = 0;				c[3][3] = 1;
	}

	// ----------
	// Matrix sum
	// ----------
	Matrix4f operator+( const Matrix4f &m ) const 
	{
		Matrix4f mf( false );

		mf.x[0] = x[0] + m.x[0];
		mf.x[1] = x[1] + m.x[1];
		mf.x[2] = x[2] + m.x[2];
		mf.x[3] = x[3] + m.x[3];
		mf.x[4] = x[4] + m.x[4];
		mf.x[5] = x[5] + m.x[5];
		mf.x[6] = x[6] + m.x[6];
		mf.x[7] = x[7] + m.x[7];
		mf.x[8] = x[8] + m.x[8];
		mf.x[9] = x[9] + m.x[9];
		mf.x[10] = x[10] + m.x[10];
		mf.x[11] = x[11] + m.x[11];
		mf.x[12] = x[12] + m.x[12];
		mf.x[13] = x[13] + m.x[13];
		mf.x[14] = x[14] + m.x[14];
		mf.x[15] = x[15] + m.x[15];

		return mf;
	}

	Matrix4f &operator+=( const Matrix4f &m )
	{
		return *this = *this + m;
	}

	// ---------------------
	// Matrix multiplication
	// ---------------------
	Matrix4f operator*( const Matrix4f &m ) const 
	{
		Matrix4f mf( false );

		mf.x[0] = x[0] * m.x[0] + x[4] * m.x[1] + x[8] * m.x[2] + x[12] * m.x[3];
		mf.x[1] = x[1] * m.x[0] + x[5] * m.x[1] + x[9] * m.x[2] + x[13] * m.x[3];
		mf.x[2] = x[2] * m.x[0] + x[6] * m.x[1] + x[10] * m.x[2] + x[14] * m.x[3];
		mf.x[3] = x[3] * m.x[0] + x[7] * m.x[1] + x[11] * m.x[2] + x[15] * m.x[3];

		mf.x[4] = x[0] * m.x[4] + x[4] * m.x[5] + x[8] * m.x[6] + x[12] * m.x[7];
		mf.x[5] = x[1] * m.x[4] + x[5] * m.x[5] + x[9] * m.x[6] + x[13] * m.x[7];
		mf.x[6] = x[2] * m.x[4] + x[6] * m.x[5] + x[10] * m.x[6] + x[14] * m.x[7];
		mf.x[7] = x[3] * m.x[4] + x[7] * m.x[5] + x[11] * m.x[6] + x[15] * m.x[7];

		mf.x[8] = x[0] * m.x[8] + x[4] * m.x[9] + x[8] * m.x[10] + x[12] * m.x[11];
		mf.x[9] = x[1] * m.x[8] + x[5] * m.x[9] + x[9] * m.x[10] + x[13] * m.x[11];
		mf.x[10] = x[2] * m.x[8] + x[6] * m.x[9] + x[10] * m.x[10] + x[14] * m.x[11];
		mf.x[11] = x[3] * m.x[8] + x[7] * m.x[9] + x[11] * m.x[10] + x[15] * m.x[11];

		mf.x[12] = x[0] * m.x[12] + x[4] * m.x[13] + x[8] * m.x[14] + x[12] * m.x[15];
		mf.x[13] = x[1] * m.x[12] + x[5] * m.x[13] + x[9] * m.x[14] + x[13] * m.x[15];
		mf.x[14] = x[2] * m.x[12] + x[6] * m.x[13] + x[10] * m.x[14] + x[14] * m.x[15];
		mf.x[15] = x[3] * m.x[12] + x[7] * m.x[13] + x[11] * m.x[14] + x[15] * m.x[15];

		return mf;
	}

	void fastMult( const Matrix4f &m1, const Matrix4f &m2 )
	{
		x[0] = m1.x[0] * m2.x[0] + m1.x[4] * m2.x[1] + m1.x[8] * m2.x[2] + m1.x[12] * m2.x[3];
		x[1] = m1.x[1] * m2.x[0] + m1.x[5] * m2.x[1] + m1.x[9] * m2.x[2] + m1.x[13] * m2.x[3];
		x[2] = m1.x[2] * m2.x[0] + m1.x[6] * m2.x[1] + m1.x[10] * m2.x[2] + m1.x[14] * m2.x[3];
		x[3] = m1.x[3] * m2.x[0] + m1.x[7] * m2.x[1] + m1.x[11] * m2.x[2] + m1.x[15] * m2.x[3];

		x[4] = m1.x[0] * m2.x[4] + m1.x[4] * m2.x[5] + m1.x[8] * m2.x[6] + m1.x[12] * m2.x[7];
		x[5] = m1.x[1] * m2.x[4] + m1.x[5] * m2.x[5] + m1.x[9] * m2.x[6] + m1.x[13] * m2.x[7];
		x[6] = m1.x[2] * m2.x[4] + m1.x[6] * m2.x[5] + m1.x[10] * m2.x[6] + m1.x[14] * m2.x[7];
		x[7] = m1.x[3] * m2.x[4] + m1.x[7] * m2.x[5] + m1.x[11] * m2.x[6] + m1.x[15] * m2.x[7];

		x[8] = m1.x[0] * m2.x[8] + m1.x[4] * m2.x[9] + m1.x[8] * m2.x[10] + m1.x[12] * m2.x[11];
		x[9] = m1.x[1] * m2.x[8] + m1.x[5] * m2.x[9] + m1.x[9] * m2.x[10] + m1.x[13] * m2.x[11];
		x[10] = m1.x[2] * m2.x[8] + m1.x[6] * m2.x[9] + m1.x[10] * m2.x[10] + m1.x[14] * m2.x[11];
		x[11] = m1.x[3] * m2.x[8] + m1.x[7] * m2.x[9] + m1.x[11] * m2.x[10] + m1.x[15] * m2.x[11];

		x[12] = m1.x[0] * m2.x[12] + m1.x[4] * m2.x[13] + m1.x[8] * m2.x[14] + m1.x[12] * m2.x[15];
		x[13] = m1.x[1] * m2.x[12] + m1.x[5] * m2.x[13] + m1.x[9] * m2.x[14] + m1.x[13] * m2.x[15];
		x[14] = m1.x[2] * m2.x[12] + m1.x[6] * m2.x[13] + m1.x[10] * m2.x[14] + m1.x[14] * m2.x[15];
		x[15] = m1.x[3] * m2.x[12] + m1.x[7] * m2.x[13] + m1.x[11] * m2.x[14] + m1.x[15] * m2.x[15];
	}

	Matrix4f operator*( const float f ) const
	{
		Matrix4f m( *this );

		for( unsigned int y = 0; y < 4; ++y )
		{
			for( unsigned int x = 0; x < 4; ++x ) 
			{
				m.c[x][y] *= f;
			}
		}

		return m;
	}

	// ----------------------------
	// Vector-Matrix multiplication
	// ----------------------------
	Vec3f operator*( const Vec3f &v ) const
	{
		return Vec3f( v.x * c[0][0] + v.y * c[1][0] + v.z * c[2][0] + c[3][0],
			v.x * c[0][1] + v.y * c[1][1] + v.z * c[2][1] + c[3][1],
			v.x * c[0][2] + v.y * c[1][2] + v.z * c[2][2] + c[3][2] );
	}

	Vec4f operator*( const Vec4f &v ) const
	{
		return Vec4f( v.x * c[0][0] + v.y * c[1][0] + v.z * c[2][0] + c[3][0],
			v.x * c[0][1] + v.y * c[1][1] + v.z * c[2][1] + c[3][1],
			v.x * c[0][2] + v.y * c[1][2] + v.z * c[2][2] + c[3][2],
			v.x * c[0][3] + v.y * c[1][3] + v.z * c[2][3] + c[3][3] );
	}

	// ---------------
	// Transformations
	// ---------------
	void translate( const float x, const float y, const float z )
	{
		*this = TransMat( x, y, z ) * *this;
	}

	void scale( const float x, const float y, const float z )
	{
		*this = ScaleMat( x, y, z ) * *this;
	}
	void scale2( const float x, const float y, const float z )
	{
		*this =  *this * ScaleMat( x, y, z );
	}

	void rotate( const float x, const float y, const float z )
	{
		*this = RotMat( x, y, z ) * *this;
	}

	// ---------------
	// Other
	// ---------------

	Matrix4f transposed() const
	{
		Matrix4f m( *this );

		for( unsigned int y = 0; y < 4; ++y )
		{
			for( unsigned int x = y + 1; x < 4; ++x ) 
			{
				float tmp = m.c[x][y];
				m.c[x][y] = m.c[y][x];
				m.c[y][x] = tmp;
			}
		}

		return m;
	}

	float determinant() const
	{
		return 
			c[0][3]*c[1][2]*c[2][1]*c[3][0] - c[0][2]*c[1][3]*c[2][1]*c[3][0] - c[0][3]*c[1][1]*c[2][2]*c[3][0] + c[0][1]*c[1][3]*c[2][2]*c[3][0] +
			c[0][2]*c[1][1]*c[2][3]*c[3][0] - c[0][1]*c[1][2]*c[2][3]*c[3][0] - c[0][3]*c[1][2]*c[2][0]*c[3][1] + c[0][2]*c[1][3]*c[2][0]*c[3][1] +
			c[0][3]*c[1][0]*c[2][2]*c[3][1] - c[0][0]*c[1][3]*c[2][2]*c[3][1] - c[0][2]*c[1][0]*c[2][3]*c[3][1] + c[0][0]*c[1][2]*c[2][3]*c[3][1] +
			c[0][3]*c[1][1]*c[2][0]*c[3][2] - c[0][1]*c[1][3]*c[2][0]*c[3][2] - c[0][3]*c[1][0]*c[2][1]*c[3][2] + c[0][0]*c[1][3]*c[2][1]*c[3][2] +
			c[0][1]*c[1][0]*c[2][3]*c[3][2] - c[0][0]*c[1][1]*c[2][3]*c[3][2] - c[0][2]*c[1][1]*c[2][0]*c[3][3] + c[0][1]*c[1][2]*c[2][0]*c[3][3] +
			c[0][2]*c[1][0]*c[2][1]*c[3][3] - c[0][0]*c[1][2]*c[2][1]*c[3][3] - c[0][1]*c[1][0]*c[2][2]*c[3][3] + c[0][0]*c[1][1]*c[2][2]*c[3][3];
	}

	Matrix4f inverted() const
	{
		Matrix4f m( false );

		float d = determinant();
		if( d == 0 ) return m;
		d = 1 / d;

		m.c[0][0] = d * (c[1][2]*c[2][3]*c[3][1] - c[1][3]*c[2][2]*c[3][1] + c[1][3]*c[2][1]*c[3][2] - c[1][1]*c[2][3]*c[3][2] - c[1][2]*c[2][1]*c[3][3] + c[1][1]*c[2][2]*c[3][3]);
		m.c[0][1] = d * (c[0][3]*c[2][2]*c[3][1] - c[0][2]*c[2][3]*c[3][1] - c[0][3]*c[2][1]*c[3][2] + c[0][1]*c[2][3]*c[3][2] + c[0][2]*c[2][1]*c[3][3] - c[0][1]*c[2][2]*c[3][3]);
		m.c[0][2] = d * (c[0][2]*c[1][3]*c[3][1] - c[0][3]*c[1][2]*c[3][1] + c[0][3]*c[1][1]*c[3][2] - c[0][1]*c[1][3]*c[3][2] - c[0][2]*c[1][1]*c[3][3] + c[0][1]*c[1][2]*c[3][3]);
		m.c[0][3] = d * (c[0][3]*c[1][2]*c[2][1] - c[0][2]*c[1][3]*c[2][1] - c[0][3]*c[1][1]*c[2][2] + c[0][1]*c[1][3]*c[2][2] + c[0][2]*c[1][1]*c[2][3] - c[0][1]*c[1][2]*c[2][3]);
		m.c[1][0] = d * (c[1][3]*c[2][2]*c[3][0] - c[1][2]*c[2][3]*c[3][0] - c[1][3]*c[2][0]*c[3][2] + c[1][0]*c[2][3]*c[3][2] + c[1][2]*c[2][0]*c[3][3] - c[1][0]*c[2][2]*c[3][3]);
		m.c[1][1] = d * (c[0][2]*c[2][3]*c[3][0] - c[0][3]*c[2][2]*c[3][0] + c[0][3]*c[2][0]*c[3][2] - c[0][0]*c[2][3]*c[3][2] - c[0][2]*c[2][0]*c[3][3] + c[0][0]*c[2][2]*c[3][3]);
		m.c[1][2] = d * (c[0][3]*c[1][2]*c[3][0] - c[0][2]*c[1][3]*c[3][0] - c[0][3]*c[1][0]*c[3][2] + c[0][0]*c[1][3]*c[3][2] + c[0][2]*c[1][0]*c[3][3] - c[0][0]*c[1][2]*c[3][3]);
		m.c[1][3] = d * (c[0][2]*c[1][3]*c[2][0] - c[0][3]*c[1][2]*c[2][0] + c[0][3]*c[1][0]*c[2][2] - c[0][0]*c[1][3]*c[2][2] - c[0][2]*c[1][0]*c[2][3] + c[0][0]*c[1][2]*c[2][3]);
		m.c[2][0] = d * (c[1][1]*c[2][3]*c[3][0] - c[1][3]*c[2][1]*c[3][0] + c[1][3]*c[2][0]*c[3][1] - c[1][0]*c[2][3]*c[3][1] - c[1][1]*c[2][0]*c[3][3] + c[1][0]*c[2][1]*c[3][3]);
		m.c[2][1] = d * (c[0][3]*c[2][1]*c[3][0] - c[0][1]*c[2][3]*c[3][0] - c[0][3]*c[2][0]*c[3][1] + c[0][0]*c[2][3]*c[3][1] + c[0][1]*c[2][0]*c[3][3] - c[0][0]*c[2][1]*c[3][3]);
		m.c[2][2] = d * (c[0][1]*c[1][3]*c[3][0] - c[0][3]*c[1][1]*c[3][0] + c[0][3]*c[1][0]*c[3][1] - c[0][0]*c[1][3]*c[3][1] - c[0][1]*c[1][0]*c[3][3] + c[0][0]*c[1][1]*c[3][3]);
		m.c[2][3] = d * (c[0][3]*c[1][1]*c[2][0] - c[0][1]*c[1][3]*c[2][0] - c[0][3]*c[1][0]*c[2][1] + c[0][0]*c[1][3]*c[2][1] + c[0][1]*c[1][0]*c[2][3] - c[0][0]*c[1][1]*c[2][3]);
		m.c[3][0] = d * (c[1][2]*c[2][1]*c[3][0] - c[1][1]*c[2][2]*c[3][0] - c[1][2]*c[2][0]*c[3][1] + c[1][0]*c[2][2]*c[3][1] + c[1][1]*c[2][0]*c[3][2] - c[1][0]*c[2][1]*c[3][2]);
		m.c[3][1] = d * (c[0][1]*c[2][2]*c[3][0] - c[0][2]*c[2][1]*c[3][0] + c[0][2]*c[2][0]*c[3][1] - c[0][0]*c[2][2]*c[3][1] - c[0][1]*c[2][0]*c[3][2] + c[0][0]*c[2][1]*c[3][2]);
		m.c[3][2] = d * (c[0][2]*c[1][1]*c[3][0] - c[0][1]*c[1][2]*c[3][0] - c[0][2]*c[1][0]*c[3][1] + c[0][0]*c[1][2]*c[3][1] + c[0][1]*c[1][0]*c[3][2] - c[0][0]*c[1][1]*c[3][2]);
		m.c[3][3] = d * (c[0][1]*c[1][2]*c[2][0] - c[0][2]*c[1][1]*c[2][0] + c[0][2]*c[1][0]*c[2][1] - c[0][0]*c[1][2]*c[2][1] - c[0][1]*c[1][0]*c[2][2] + c[0][0]*c[1][1]*c[2][2]);

		return m;
	}

	void decompose( Vec3f &trans, Vec3f &rot, Vec3f &scale ) const
	{
		// Getting translation is trivial
		trans = Vec3f( c[3][0], c[3][1], c[3][2] );

		// Scale is length of columns
		scale.x = sqrt( c[0][0] * c[0][0] + c[0][1] * c[0][1] + c[0][2] * c[0][2] );
		scale.y = sqrt( c[1][0] * c[1][0] + c[1][1] * c[1][1] + c[1][2] * c[1][2] );
		scale.z = sqrt( c[2][0] * c[2][0] + c[2][1] * c[2][1] + c[2][2] * c[2][2] );

		if( scale.x == 0 || scale.y == 0 || scale.z == 0 ) return;

		// Detect negative scale with determinant and flip one arbitrary axis
		if( determinant() < 0 ) scale.x = -scale.x;

		// Combined rotation matrix YXZ
		//
		// Cos[y]*Cos[z]+Sin[x]*Sin[y]*Sin[z]	Cos[z]*Sin[x]*Sin[y]-Cos[y]*Sin[z]	Cos[x]*Sin[y]	
		// Cos[x]*Sin[z]						Cos[x]*Cos[z]						-Sin[x]
		// -Cos[z]*Sin[y]+Cos[y]*Sin[x]*Sin[z]	Cos[y]*Cos[z]*Sin[x]+Sin[y]*Sin[z]	Cos[x]*Cos[y]

		rot.x = asinf( -c[2][1] / scale.z );

		// Special case: Cos[x] == 0 (when Sin[x] is +/-1)
		float f = fabsf( c[2][1] / scale.z );
		if( f > 0.999f && f < 1.001f )
		{
			// Pin arbitrarily one of y or z to zero
			// Mathematical equivalent of gimbal lock
			rot.y = 0;

			// Now: Cos[x] = 0, Sin[x] = +/-1, Cos[y] = 1, Sin[y] = 0
			// => m[0][0] = Cos[z] and m[1][0] = Sin[z]
			rot.z = atan2f( -c[1][0] / scale.y, c[0][0] / scale.x );
		}
		// Standard case
		else
		{
			rot.y = atan2f( c[2][0] / scale.z, c[2][2] / scale.z );
			rot.z = atan2f( c[0][1] / scale.x, c[1][1] / scale.y );
		}
	}

	Vec4f getCol( unsigned int col ) const
	{
		return Vec4f( x[col * 4 + 0], x[col * 4 + 1], x[col * 4 + 2], x[col * 4 + 3] );
	}

	Vec4f getRow( unsigned int row ) const
	{
		return Vec4f( x[row + 0], x[row + 4], x[row + 8], x[row + 12] );
	}
};


// -------------------------------------------------------------------------------------------------
// Plane
// -------------------------------------------------------------------------------------------------

class Plane
{
public:
	Vec3f normal; 
	float dist;

	// ------------
	// Constructors
	// ------------
	Plane() 
	{ 
		normal.x = 0; normal.y = 0; normal.z = 0; dist = 0; 
	};

	explicit Plane( const float a, const float b, const float c, const float d )
	{
		normal = Vec3f( a, b, c );
		float len = normal.length();
		normal /= len;	// Normalize
		dist = d / len;
	}

	Plane( const Vec3f &v0, const Vec3f &v1, const Vec3f &v2 )
	{
		normal = v1 - v0;
		normal = normal.crossProduct( v2 - v0 ).normalized();
		dist = -(normal * v0);
	}

	// ----------------
	// Other operations
	// ----------------
	float distToPoint( const Vec3f &v ) const
	{
		return (normal * v) + dist;
	}
};


// -------------------------------------------------------------------------------------------------
// Intersection
// -------------------------------------------------------------------------------------------------

inline bool rayTriangleIntersection( const Vec3f &rayOrig, const Vec3f &rayDir, 
									const Vec3f &vert0, const Vec3f &vert1, const Vec3f &vert2,
									Vec3f &intsPoint )
{
	// Idea: Tomas Moeller and Ben Trumbore
	// in Fast, Minimum Storage Ray/Triangle Intersection 

	// Find vectors for two edges sharing vert0
	Vec3f edge1 = vert1 - vert0;
	Vec3f edge2 = vert2 - vert0;

	// Begin calculating determinant - also used to calculate U parameter
	Vec3f pvec = rayDir.crossProduct( edge2 );

	// If determinant is near zero, ray lies in plane of triangle
	float det = edge1 * pvec;


	// *** Culling branch ***
	/*if( det < Math::Epsilon )return false;

	// Calculate distance from vert0 to ray origin
	Vec3f tvec = rayOrig - vert0;

	// Calculate U parameter and test bounds
	float u = tvec * pvec;
	if (u < 0 || u > det ) return false;

	// Prepare to test V parameter
	Vec3f qvec = tvec.crossProduct( edge1 );

	// Calculate V parameter and test bounds
	float v = rayDir * qvec;
	if (v < 0 || u + v > det ) return false;

	// Calculate t, scale parameters, ray intersects triangle
	float t = (edge2 * qvec) / det;*/


	// *** Non-culling branch ***
	if( det > -Math::Epsilon && det < Math::Epsilon ) return 0;
	float inv_det = 1.0f / det;

	// Calculate distance from vert0 to ray origin
	Vec3f tvec = rayOrig - vert0;

	// Calculate U parameter and test bounds
	float u = (tvec * pvec) * inv_det;
	if( u < 0.0f || u > 1.0f ) return 0;

	// Prepare to test V parameter
	Vec3f qvec = tvec.crossProduct( edge1 );

	// Calculate V parameter and test bounds
	float v = (rayDir * qvec) * inv_det;
	if( v < 0.0f || u + v > 1.0f ) return 0;

	// Calculate t, ray intersects triangle
	float t = (edge2 * qvec) * inv_det;


	// Calculate intersection point and test ray length and direction
	intsPoint = rayOrig + rayDir * t;
	Vec3f vec = intsPoint - rayOrig;
	if( vec * rayDir < 0 || vec.length() > rayDir.length() ) return false;

	return true;
}


inline bool rayAABBIntersection( const Vec3f &rayOrig, const Vec3f &rayDir, 
								const Vec3f &mins, const Vec3f &maxs )
{
	// TODO: This routine considers only ray direction and not length

	// SLAB based optimized ray/AABB intersection routine
	// Idea taken from http://ompf.org/ray/

	float l1 = (mins.x - rayOrig.x) / rayDir.x;
	float l2 = (maxs.x - rayOrig.x) / rayDir.x;
	float lmin = minf( l1, l2 );
	float lmax = maxf( l1, l2 );

	l1 = (mins.y - rayOrig.y) / rayDir.y;
	l2 = (maxs.y - rayOrig.y) / rayDir.y;
	lmin = maxf( minf( l1, l2 ), lmin );
	lmax = minf( maxf( l1, l2 ), lmax );

	l1 = (mins.z - rayOrig.z) / rayDir.z;
	l2 = (maxs.z - rayOrig.z) / rayDir.z;
	lmin = maxf( minf( l1, l2 ), lmin );
	lmax = minf( maxf( l1, l2 ), lmax );

	return (lmax >= 0.0f) & (lmax >= lmin);
}


inline float nearestDistToAABB( const Vec3f &pos, const Vec3f &mins, const Vec3f &maxs )
{
	const Vec3f center = (mins + maxs) / 2.0f;
	const Vec3f extent = (maxs - mins) / 2.0f;

	Vec3f nearestVec;
	nearestVec.x = maxf( 0, fabsf( pos.x - center.x ) - extent.x );
	nearestVec.y = maxf( 0, fabsf( pos.y - center.y ) - extent.y );
	nearestVec.z = maxf( 0, fabsf( pos.z - center.z ) - extent.z );

	return nearestVec.length();
}