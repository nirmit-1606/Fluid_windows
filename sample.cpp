#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>
#include <iomanip> // for std::setprecision
#include <omp.h>  // Include OpenMP header

#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>


#ifndef F_PI
#define F_PI		((float)(M_PI))
#define F_2_PI		((float)(2.f*F_PI))
#define F_PI_2		((float)(F_PI/2.f))
#endif


#ifdef WIN32
#include <windows.h>
#pragma warning(disable:4996)
#endif


#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include "glew.h"
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "glut.h"

#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

//	This is a sample OpenGL / GLUT program
//
//	The objective is to draw a 3d object and change the color of the axes
//		with a glut menu
//
//	The left mouse button does rotation
//	The middle mouse button does scaling
//	The user interface allows:
//		1. The axes to be turned on and off
//		2. The color of the axes to be changed
//		3. Debugging to be turned on and off
//		4. Depth cueing to be turned on and off
//		5. The projection to be changed
//		6. The transformations to be reset
//		7. The program to quit
//
//	Author for loading the window:			Joe Graphics
//  Author for implementing simulation: 	Nirmit Patel

// title of these windows:

const char* WINDOWTITLE = "OpenGL / GLUT Simulation -- Nirmit Patel";
const char* GLUITITLE = "User Interface Window";

// what the glui package defines as true and false:

const int GLUITRUE = true;
const int GLUIFALSE = false;

// the escape key:

const int ESCAPE = 0x1b;

// initial window size:

const int INIT_WINDOW_SIZE = 600;

// multiplication factors for input interaction:
//  (these are known from previous experience)

const float ANGFACT = 1.f;
const float SCLFACT = 0.005f;

// minimum allowable scale factor:

const float MINSCALE = 0.05f;

// scroll wheel button values:

const int SCROLL_WHEEL_UP = 3;
const int SCROLL_WHEEL_DOWN = 4;

// equivalent mouse movement when we click the scroll wheel:

const float SCROLL_WHEEL_CLICK_FACTOR = 5.f;

// active mouse buttons (or them together):

const int LEFT = 4;
const int MIDDLE = 2;
const int RIGHT = 1;

// which projection:

enum Projections
{
	ORTHO,
	PERSP
};

// which button:

enum ButtonVals
{
	RESET,
	QUIT
};

// window background color (rgba):

const GLfloat BACKCOLOR[] = { 0., 0., 0., 1. };

// line width for the axes:

const GLfloat AXES_WIDTH = 3.;

// the color numbers:
// this order must match the radio button order, which must match the order of the color names,
// 	which must match the order of the color RGB values

enum Colors
{
	RED,
	YELLOW,
	GREEN,
	CYAN,
	BLUE,
	MAGENTA
};

char* ColorNames[] =
{
	(char*)"Red",
	(char*)"Yellow",
	(char*)"Green",
	(char*)"Cyan",
	(char*)"Blue",
	(char*)"Magenta" };

// the color definitions:
// this order must match the menu order

const GLfloat Colors[][3] =
{
	{1., 0., 0.}, // red
	{1., 1., 0.}, // yellow
	{0., 1., 0.}, // green
	{0., 1., 1.}, // cyan
	{0., 0., 1.}, // blue
	{1., 0., 1.}, // magenta
};

// fog parameters:

const GLfloat FOGCOLOR[4] = { .0f, .0f, .0f, 1.f };
const GLenum FOGMODE = GL_LINEAR;
const GLfloat FOGDENSITY = 0.30f;
const GLfloat FOGSTART = 1.5f;
const GLfloat FOGEND = 4.f;

// for lighting:

const float WHITE[] = { 1., 1., 1., 1. };

// for animation:

const int MS_PER_CYCLE = 30000; // 10000 milliseconds = 10 seconds

// --------------------------------------------------------------------
// A structure for holding two neighboring particles and their weighted distances
struct Particle;
struct Neighbor
{
	Particle* j;
	float q, q2;
};

// --------------------------------------------------------------------
// The Particle structure holding all of the relevant information.
struct Particle
{
	glm::vec3 pos;
	float r, g, b;

	glm::vec3 pos_old;
	glm::vec3 vel;
	glm::vec3 force;
	float mass;
	float rho;
	float rho_near;
	float press;
	float press_near;
	float sigma;
	float beta;
	std::vector<Neighbor> neighbors;
};

// Our collection of particles
std::vector<Particle> particles;

// --------------------------------------------------------------------
// Some constants for the relevant simulation.
const float p_size = 4;		   // particle size
const float dT = 0.01;			// delta time, for step iteration
const float sigma = 1.;
const float beta = 1.;
const float G = .001f * .25;		   // Gravitational Constant for our simulation
const float spacing = .07f;		   // Spacing of particles
const float k = spacing / 1000.0f; // Far pressure weight
const float k_near = k * 10.;	   // Near pressure weight
const float r = spacing * 1.25f;   // Radius of Support
const float rsq = r * r;		   // ... squared for performance stuff
const float SIM_W = .8;		   // The size of the world
const float bottom = 0;			   // The floor of the world
const float i_girth = 1.f;		   // initial parameters

int N = 500;
float rest_density = 3.;	   // Rest Density

// #define DEMO_Z_FIGHTING
// #define DEMO_DEPTH_BUFFER

// non-constant global variables:

int ActiveButton;	 // current button that is down
GLuint AxesList;	 // list to hold the axes
int AxesOn;			 // != 0 means to draw the axes
GLuint ParticleList; // object display list
GLuint GridDL1;		 // object display list
GLuint GridDL2;		 // object display list
int DebugOn;		 // != 0 means to print debugging info
int DepthCueOn;		 // != 0 means to use intensity depth cueing
int DepthBufferOn;	 // != 0 means to use the z-buffer
int DepthFightingOn; // != 0 means to force the creation of z-fighting
int MainWindow;		 // window id for main graphics window
int NowColor;		 // index into Colors[ ]
int NowProjection;	 // ORTHO or PERSP
float Scale;		 // scaling factor
int ShadowsOn;		 // != 0 means to turn shadows on
float Time;			 // used for animation, this has a value between 0. and 1.
int Xmouse, Ymouse;	 // mouse values
float Xrot, Yrot;	 // rotation angles in degrees
int displayCnt;
double timeSum;

bool doSimulation;
bool useGravity;
bool useColorVisual;
bool externalForce;
bool shrinkWorld;
bool useLighting;
bool useOpening;

// function prototypes:

void Animate();
void Display();
void DoAxesMenu(int);
void DoColorMenu(int);
void DoDepthBufferMenu(int);
void DoDepthFightingMenu(int);
void DoDepthMenu(int);
void DoDebugMenu(int);
void DoMainMenu(int);
void DoProjectMenu(int);
void DoRasterString(float, float, float, char*);
void DoStrokeString(float, float, float, float, char*);
float ElapsedSeconds();
void InitGraphics();
void InitLists();
void InitMenus();
void Keyboard(unsigned char, int, int);
void MouseButton(int, int, int, int);
void MouseMotion(int, int);
void Reset();
void Resize(int, int);
void Visibility(int);

void Axes(float);
void HsvRgb(float[3], float[3]);
void Cross(float[3], float[3], float[3]);
float Dot(float[3], float[3]);
float Unit(float[3], float[3]);
float Unit(float[3]);

void initParticles(const unsigned int);
void addMoreParticles(const unsigned int);
void step();

// utility to create an array from 3 separate values:

float*
Array3(float a, float b, float c)
{
	static float array[4];

	array[0] = a;
	array[1] = b;
	array[2] = c;
	array[3] = 1.;
	return array;
}

float*
Array4(float a, float b, float c, float d)
{
	static float array[4];

	array[0] = a;
	array[1] = b;
	array[2] = c;
	array[3] = d;
	return array;
}

// utility to create an array from a multiplier and an array:

float*
MulArray3(float factor, float array0[])
{
	static float array[4];

	array[0] = factor * array0[0];
	array[1] = factor * array0[1];
	array[2] = factor * array0[2];
	array[3] = 1.;
	return array;
}

float*
MulArray3(float factor, float a, float b, float c)
{
	static float array[4];

	float* abc = Array3(a, b, c);
	array[0] = factor * abc[0];
	array[1] = factor * abc[1];
	array[2] = factor * abc[2];
	array[3] = 1.;
	return array;
}

// --------------------------------------------------------------------
// Between [0,1]
float rand01()
{
	return (float)rand() * (1.f / RAND_MAX);
}

// --------------------------------------------------------------------
// Between [a,b]
float randab(float a, float b)
{
	return a + (b - a) * rand01();
}

// these are here for when you need them -- just uncomment the ones you need:

#include "setmaterial.cpp"
#include "setlight.cpp"
#include "osusphere.cpp"
// #include "osucone.cpp"
// #include "osutorus.cpp"
// #include "bmptotexture.cpp"
// #include "loadobjfile.cpp"
// #include "keytime.cpp"
// #include "glslprogram.cpp"'

// --------------------------------------------------------------------
template <typename T>
class SpatialIndex
{
public:
	typedef std::vector<T*> NeighborList;

	SpatialIndex(
		const unsigned int numBuckets, // number of hash buckets
		const float cellSize		   // grid cell size
	)
		: mHashMap(numBuckets), mInvCellSize(1.0f / cellSize)
	{
		// initialize neighbor offsets
		for (int i = -1; i <= 1; i++)
			for (int j = -1; j <= 1; j++)
				for (int k = -1; k <= 1; k++)
					mOffsets.push_back(glm::ivec3(i, j, k));
	}

	void Insert(const glm::vec3& pos, T* thing)
	{
		mHashMap[Discretize(pos, mInvCellSize)].push_back(thing);
	}

	void Neighbors(const glm::vec3& pos, NeighborList& ret) const
	{
		const glm::ivec3 ipos = Discretize(pos, mInvCellSize);
		for (const auto& offset : mOffsets)
		{
			typename HashMap::const_iterator it = mHashMap.find(offset + ipos);
			if (it != mHashMap.end())
			{
				ret.insert(ret.end(), it->second.begin(), it->second.end());
			}
		}
	}

	void Clear()
	{
		mHashMap.clear();
	}

private:
	// "Optimized Spatial Hashing for Collision Detection of Deformable Objects"
	// Teschner, Heidelberger, et al.
	// returns a hash between 0 and 2^32-1
	struct TeschnerHash{
		std::size_t operator()(const glm::ivec3& pos) const
		{
			const unsigned int p1 = 73856093;
			const unsigned int p2 = 19349663;
			const unsigned int p3 = 83492791;
			return static_cast<std::size_t>((pos.x * p1) ^ (pos.y * p2) ^ (pos.z * p3));
		}
	};

	// returns the indexes of the cell pos is in, assuming a cellSize grid
	// invCellSize is the inverse of the desired cell size
	static inline glm::ivec3 Discretize(const glm::vec3& pos, const float invCellSize)
	{
		return glm::ivec3(glm::floor(pos * invCellSize));
	}

	typedef std::unordered_map<glm::ivec3, NeighborList, TeschnerHash> HashMap;
	HashMap mHashMap;

	std::vector<glm::ivec3> mOffsets;

	const float mInvCellSize;
};

typedef SpatialIndex<Particle> IndexType;
IndexType indexsp(4093, r * 2);

// --------------------------------------------------------------------
void initParticles(const unsigned int pN)
{
	float layer_radius = i_girth * 0.2;  // Radius of the cylindrical layer
	float maxHeight = 5.0;               // Maximum height of the cylinder
	float minDistance = r * 0.5f;        // Minimum distance between particles

	for (float y = bottom + 0.1; y <= maxHeight; y += minDistance)
	{
		// Start from the center and place particles in concentric rings
		for (float radius = 0; radius <= layer_radius; radius += minDistance)
		{
			// Number of particles around this radius (circumference / min distance)
			int numParticles = (radius == 0) ? 1 : static_cast<int>((2 * M_PI * radius) / minDistance);

			for (int i = 0; i < numParticles; ++i)
			{
				if (particles.size() >= pN)  // Stop if we reach the desired number of particles
				{
					return;
				}

				// Angle for this particle in the current ring
				float angle = i * (2 * M_PI / numParticles);

				// Convert polar coordinates (radius, angle) to Cartesian (x, z)
				float x = radius * cos(angle);
				float z = radius * sin(angle);

				Particle p;
				p.pos = glm::vec3(x, y, z) + 0.01f * glm::vec3(rand01(), rand01(), rand01());
				p.pos_old = p.pos + 0.001f * glm::vec3(rand01(), rand01(), rand01());
				p.vel = glm::vec3(0, 0, 0);
				p.force = glm::vec3(0, 0, 0);
				p.sigma = 3.f;
				p.beta = 4.f;
				particles.push_back(p);
			}
		}
	}
}

void addMoreParticles(const unsigned int nP)
{
	// Number of particles already in the system
	unsigned int currentParticleCount = particles.size();

	float layer_radius = i_girth * 0.2;  // Radius of the cylindrical layer
	float maxHeight = 5.0;               // Maximum height of the cylinder
	float minDistance = r * 0.5f;        // Minimum distance between particles

	for (float y = bottom + 1.8; y <= maxHeight; y += minDistance)
	{
		// Start from the center and place particles in concentric rings
		for (float radius = 0; radius <= layer_radius; radius += minDistance)
		{
			// Number of particles around this radius (circumference / min distance)
			int numParticles = (radius == 0) ? 1 : static_cast<int>((2 * M_PI * radius) / minDistance);

			for (int i = 0; i < numParticles; ++i)
			{
				// Only add new particles up to the specified pN
				if (particles.size() >= currentParticleCount + nP)
				{
					break;
				}

				// Angle for this particle in the current ring
				float angle = i * (2 * M_PI / numParticles);

				// Convert polar coordinates (radius, angle) to Cartesian (x, z)
				float x = radius * cos(angle);
				float z = radius * sin(angle);

				Particle p;
				p.pos = glm::vec3(x, y, z) + 0.01f * glm::vec3(rand01(), rand01(), rand01());
				p.pos_old = p.pos + 0.001f * glm::vec3(rand01(), rand01(), rand01());
				p.vel = glm::vec3(0, 0, 0);
				p.force = glm::vec3(0, 0, 0);
				p.sigma = 3.f;
				p.beta = 4.f;
				particles.push_back(p);
			}
		}
	}
}

// Define container properties
const float container_height = 1.5f;    // Height of the container bottom
const float container_top = container_height + 2.0f; // Top boundary of the container
const float container_width = SIM_W / 2;     // Width of the container in the x and z directions
const float opening_width = 0.2f;       // Width of the opening at the container's bottom

void enforceContainerBoundaries(Particle& p) {
	// Left and right walls in x-direction (container boundaries)
	if (p.pos.x < -container_width) {
		p.force.x -= (p.pos.x + container_width) / 8;
	}
	if (p.pos.x > container_width) {
		p.force.x -= (p.pos.x - container_width) / 8;
	}

	// Front and back walls in z-direction (container boundaries)
	if (p.pos.z < -container_width) {
		p.force.z -= (p.pos.z + container_width) / 8;
	}
	if (p.pos.z > container_width) {
		p.force.z -= (p.pos.z - container_width) / 8;
	}

	// Bottom boundary of the container, excluding the opening
	if (p.pos.y < container_height &&
		(!useOpening || (abs(p.pos.x) > opening_width / 2 || abs(p.pos.z) > opening_width / 2)))
	{
		// Only apply force if particle is outside the opening
		p.force.y -= (p.pos.y - container_height) / 8;
	}

	// Top boundary of the container
	if (p.pos.y > container_top) {
		p.force.y -= (p.pos.y - container_top) / 8;
	}
}


// --------------------------------------------------------------------
// Update particle positions
void step()
{
	// Simulation step

#pragma omp parallel for
	for (int i = 0; i < (int)particles.size(); ++i)
	{
		// Apply the currently accumulated forces
		particles[i].pos += particles[i].force;

		// Restart the forces with gravity only. We'll add the rest later.
		if (useGravity)
		{
			particles[i].force = glm::vec3(0.f, -::G, 0.f);
		}
		else
		{
			particles[i].force = glm::vec3(0.f, 0.f, 0.f);
		}

		// Calculate the velocity for later.
		particles[i].vel = particles[i].pos - particles[i].pos_old;

		// A small hack
		const float max_vel = 2.0f;
		const float vel_mag = glm::dot(particles[i].vel, particles[i].vel);
		// If the velocity is greater than the max velocity, then cut it in half.
		if (vel_mag > max_vel * max_vel)
		{
			particles[i].vel *= .08f;
		}

		// Normal verlet stuff
		particles[i].pos_old = particles[i].pos;
		particles[i].pos += particles[i].vel;

		// If the Particle is outside the bounds of the world, then
		// Make a little spring force to push it back in.
		if (useGravity)
		{
			if (particles[i].pos.y >= container_height - 0.05)
				enforceContainerBoundaries(particles[i]);
			else {
				float bound;
				if (shrinkWorld)
				{
					bound = SIM_W;
				}
				else
				{
					bound = SIM_W * 3.f;
				}
				// // Calculate the distance of the particle from the circle center in the xz-plane
				// float dx = particles[i].pos.x - 0.f; // center_x = 0
				// float dz = particles[i].pos.z - 0.f; // center_z = 0
				// float distance_from_center = sqrt(dx * dx + dz * dz);

				// // If the particle is outside the circular boundary
				// if (distance_from_center > bound) {
				// 	// Calculate the push-back force
				// 	float excess_distance = distance_from_center - bound;

				// 	// Normalize the direction vector (dx, dz)
				// 	float nx = dx / distance_from_center;
				// 	float nz = dz / distance_from_center;

				// 	// Apply force to push the particle back within the circle
				// 	particles[i].force.x -= nx * excess_distance / 8;
				// 	particles[i].force.z -= nz * excess_distance / 8;
				// }

				if (particles[i].pos.x < -bound)
					particles[i].force.x -= (particles[i].pos.x - -bound) / 8;
				if (particles[i].pos.x > bound)
					particles[i].force.x -= (particles[i].pos.x - bound) / 8;

				if (particles[i].pos.z < -SIM_W)
					particles[i].force.z -= (particles[i].pos.z - -SIM_W) / 8;
				if (particles[i].pos.z > SIM_W)
					particles[i].force.z -= (particles[i].pos.z - SIM_W) / 8;

				// Limit particles in y-axis (for bottom boundary)
				if (particles[i].pos.y < bottom) {
					particles[i].force.y -= (particles[i].pos.y) / 8;
				}

				// if (particles[i].pos.y > bottom+10)
				// 	particles[i].force.y -= (particles[i].pos.y - (bottom+10)) / 8;
			}
		}

		if (externalForce)
		{
			particles[i].force += glm::vec3(.002f * 0.025, 0.f, 0.f);
		}

		// Reset the nessecary items.
		// particles[i].rho = 0;
		// particles[i].rho_near = 0;
		particles[i].neighbors.clear();
	}

	// update spatial index
	indexsp.Clear();
	for (auto& particle : particles)
	{
		indexsp.Insert(glm::vec3(particle.pos), &particle);
	}

	// DENSITY
	// Calculate the density by basically making a weighted sum
	// of the distances of neighboring particles within the radius of support (r)
#pragma omp parallel for
	for (int i = 0; i < (int)particles.size(); ++i)
	{
		particles[i].rho = 0;
		particles[i].rho_near = 0;

		// We will sum up the 'near' and 'far' densities.
		float d = 0;
		float dn = 0;

		IndexType::NeighborList neigh;
		neigh.reserve(64);		// 64 original
		indexsp.Neighbors(glm::vec3(particles[i].pos), neigh);
		for (int j = 0; j < (int)neigh.size(); ++j)
		{
			if (neigh[j] == &particles[i])
			{
				// do not calculate an interaction for a Particle with itself!
				continue;
			}

			// The vector seperating the two particles
			const glm::vec3 rij = neigh[j]->pos - particles[i].pos;

			// Along with the squared distance between
			const float rij_len2 = glm::dot(rij, rij);

			// If they're within the radius of support ...
			if (rij_len2 < rsq)
			{
				// Get the actual distance from the squared distance.
				float rij_len = sqrt(rij_len2);

				// And calculated the weighted distance values
				const float q = 1 - (rij_len / r);
				const float q2 = q * q;
				const float q3 = q2 * q;

				d += q2;
				dn += q3;

				// Set up the Neighbor list for faster access later.
				Neighbor n;
				n.j = neigh[j];
				n.q = q;
				n.q2 = q2;
				particles[i].neighbors.push_back(n);
			}
		}

		particles[i].rho += d;
		particles[i].rho_near += dn;
	}

	// PRESSURE
	// Make the simple pressure calculation from the equation of state.
#pragma omp parallel for
	for (int i = 0; i < (int)particles.size(); ++i)
	{
		particles[i].press = k * (particles[i].rho - rest_density);
		particles[i].press_near = k_near * particles[i].rho_near;
	}

	// PRESSURE FORCE
	// We will force particles in or out from their neighbors
	// based on their difference from the rest density.
#pragma omp parallel for
	for (int i = 0; i < (int)particles.size(); ++i)
	{
		// For each of the neighbors
		glm::vec3 dX(0);
		for (const Neighbor& n : particles[i].neighbors)
		{
			// The vector from Particle i to Particle j
			const glm::vec3 rij = (*n.j).pos - particles[i].pos;

			// calculate the force from the pressures calculated above
			const float dm = n.q * (particles[i].press + (*n.j).press) + n.q2 * (particles[i].press_near + (*n.j).press_near);

			// Get the direction of the force
			const glm::vec3 D = glm::normalize(rij) * dm;
			dX += D;
		}

#pragma omp critical
		{
			particles[i].force -= dX;
		}
	}

	// Viscosity
	// fprintf(stderr, "rho: %3.5f\n", 80000.f * fabs(glm::dot(particles[0].vel.x, particles[0].vel.z)));
#pragma omp parallel for
	for (int i = 0; i < (int)particles.size(); ++i)
	{
		// We'll let the color be determined by
		// ... x-velocity for the red component
		// ... y-velocity for the green-component
		// ... pressure for the blue component
		particles[i].r = 0.3f + (80000.f * fabs(glm::dot(particles[i].vel.x, particles[i].vel.z)));
		particles[i].g = 0.3f + (60.f * fabs(particles[i].vel.y));
		particles[i].b = 0.3f + (.6f * particles[i].rho);

		// For each of that particles neighbors
		for (const Neighbor& n : particles[i].neighbors)
		{
			const glm::vec3 rij = (*n.j).pos - particles[i].pos;
			const float l = glm::length(rij);
			const float q = l / r;

			const glm::vec3 rijn = (rij / l);
			// Get the projection of the velocities onto the vector between them.
			const float u = glm::dot(particles[i].vel - (*n.j).vel, rijn);
			if (u > 0)
			{
				// Calculate the viscosity impulse between the two particles
				// based on the quadratic function of projected length.
				const glm::vec3 I = (1 - q) * ((*n.j).sigma * u + (*n.j).beta * u * u) * rijn;

				// Apply the impulses on the current particle
				particles[i].vel -= I * 0.5f;
			}
		}
	}
}

// main program:

int main(int argc, char* argv[])
{
	// turn on the glut package:
	// (do this before checking argc and argv since glutInit might
	// pull some command line arguments out)

#ifdef _OPENMP
	// fprintf( stderr, "OpenMP version %d is supported here\n", _OPENMP );
#else
	fprintf(stderr, "OpenMP is not supported here - sorry!\n");
	exit(0);
#endif

	int numprocs = omp_get_num_procs();
	// fprintf( stderr, "Number of cores present in the system: %d\n", numprocs );

	omp_set_num_threads(numprocs);

	glutInit(&argc, argv);

	// setup all the graphics stuff:

	InitGraphics();

	// create the display lists that **will not change**:

	InitLists();

	// init all the global variables used by Display( ):
	// this will also post a redisplay

	Reset();

	// Initialize initial number of particles
	initParticles(N);

	// setup all the user interface stuff:

	InitMenus();

	// draw the scene once and wait for some interaction:
	// (this will never return)

	glutSetWindow(MainWindow);
	glutMainLoop();

	// glutMainLoop( ) never actually returns
	// the following line is here to make the compiler happy:

	return 0;
}

// this is where one would put code that is to be called
// everytime the glut main loop has nothing to do
//
// this is typically where animation parameters are set
//
// do not call Display( ) from here -- let glutPostRedisplay( ) do it

void Animate()
{
	// put animation stuff in here -- change some global variables for Display( ) to find:

	// int ms = glutGet(GLUT_ELAPSED_TIME);
	// ms %= MS_PER_CYCLE;						// makes the value of ms between 0 and MS_PER_CYCLE-1
	// Time = (float)ms / (float)MS_PER_CYCLE; // makes the value of Time between 0. and slightly less than 1.
	// fprintf(stderr, "%f\n", Time);

	// for example, if you wanted to spin an object in Display( ), you might call: glRotatef( 360.f*Time,   0., 1., 0. );

	// force a call to Display( ) next time it is convenient:

	glutSetWindow(MainWindow);
	glutPostRedisplay();
}

// draw the complete scene:
void Display()
{
	if (DebugOn != 0)
		fprintf(stderr, "Starting Display.\n");

	// set which window we want to do the graphics into:
	glutSetWindow(MainWindow);

	// erase the background:
	// glDrawBuffer(GL_BACK);
	glClearColor(.19f, .28f, .28f, 1.f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
#ifdef DEMO_DEPTH_BUFFER
	if (DepthBufferOn == 0)
		glDisable(GL_DEPTH_TEST);
#endif

	// specify shading to be flat:

	glShadeModel(GL_SMOOTH);

	// set the viewport to be a square centered in the window:

	GLsizei vx = glutGet(GLUT_WINDOW_WIDTH);
	GLsizei vy = glutGet(GLUT_WINDOW_HEIGHT);
	GLsizei v = vx < vy ? vx : vy; // minimum dimension
	GLint xl = (vx - v) / 2;
	GLint yb = (vy - v) / 2;
	glViewport(xl, yb, v, v);

	// set the viewing volume:
	// remember that the Z clipping  values are given as DISTANCES IN FRONT OF THE EYE
	// USE gluOrtho2D( ) IF YOU ARE DOING 2D !

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	if (NowProjection == ORTHO)
		glOrtho(-2.f, 2.f, -2.f, 2.f, 0.1f, 1000.f);
	else
		gluPerspective(70.f, 1.f, 0.1f, 1000.f);

	// place the objects into the scene:

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// set the eye position, look-at position, and up-vector:

	gluLookAt(0.f, 5.f, 5.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f);

	// rotate the scene:

	glRotatef((GLfloat)Yrot, 0.f, 1.f, 0.f);
	glRotatef((GLfloat)Xrot, 1.f, 0.f, 0.f);

	// uniformly scale the scene:

	if (Scale < MINSCALE)
		Scale = MINSCALE;
	glScalef((GLfloat)Scale, (GLfloat)Scale, (GLfloat)Scale);

	// set the fog parameters:

	if (DepthCueOn != 0)
	{
		glFogi(GL_FOG_MODE, FOGMODE);
		glFogfv(GL_FOG_COLOR, FOGCOLOR);
		glFogf(GL_FOG_DENSITY, FOGDENSITY);
		glFogf(GL_FOG_START, FOGSTART);
		glFogf(GL_FOG_END, FOGEND);
		glEnable(GL_FOG);
	}
	else
	{
		glDisable(GL_FOG);
	}

	// possibly draw the axes:

	// if (AxesOn != 0)
	// {
	// 	glColor3fv(&Colors[NowColor][0]);
	// 	glCallList(AxesList);
	// }

	// since we are using glScalef( ), be sure the normals get unitized:

	glEnable(GL_NORMALIZE);

	// glPointSize(p_size);
	// // SetMaterial(.2, .9, 1., 10.);

	// // Enable vertex arrays for positions
	// glVertexPointer(3, GL_FLOAT, sizeof(Particle), &particles[0].pos);
	// glEnableClientState(GL_VERTEX_ARRAY);

	// // Prepare and enable color arrays for particles
	// std::vector<float> particleColors;
	// particleColors.reserve(particles.size() * 3);  // r, g, b for each particle

	// for (const auto& particle : particles) {
	// 	particleColors.push_back(particle.r); // red component
	// 	particleColors.push_back(particle.g); // green component
	// 	particleColors.push_back(particle.b); // blue component
	// }

	// glColor3f(.5, .6, .9);

	// // Use the color array
	// glColorPointer(3, GL_FLOAT, 0, particleColors.data());
	// if (useColorVisual)
	// {
	// 	glEnableClientState(GL_COLOR_ARRAY);
	// }

	// // Draw particles
	// glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(particles.size()));

	// // Disable arrays after drawing
	// glDisableClientState(GL_VERTEX_ARRAY);
	// glDisableClientState(GL_COLOR_ARRAY);

	if (useLighting)
	{
		glEnable(GL_LIGHTING);
		SetPointLight(GL_LIGHT0, -5., 5., 5., 1., 1., 1.);
	}

	// Iterate through your particles and draw spheres at their positions
	for (const auto& particle : particles)
	{
		if (useColorVisual)
		{
			glColor3f(particle.r, particle.g, particle.b);
		}
		else
		{
			glColor3f(.2, .9, 1.);
		}
		if (useLighting)
		{
			SetMaterial(particle.r, particle.g, particle.b, 5.);
		}
		glPushMatrix();
		glTranslatef(particle.pos.x, particle.pos.y, particle.pos.z); // Translate to the position of the particle
		glCallList(ParticleList);									  // Draw low-poly sphere at the position
		glPopMatrix();
	}

	if (useGravity)
	{
		glColor3f(.1, .2, .3);
		if (shrinkWorld)
			glCallList(GridDL1);
		else
			glCallList(GridDL2);
	}

	if (doSimulation) {
		double time0 = omp_get_wtime();	// current clock time in seconds
		step();
		double time1 = omp_get_wtime();	// current clock time in seconds
		// displayCnt++;
		if (displayCnt < 50)
		{
			// printf("time0 %.2f, time1 %.2f\n", time0, time1);
			displayCnt++;
			double t_diff = (time1 - time0) * 1000000.;
			timeSum += t_diff;
			// printf("For %lu partiles, time to compute is: %.2f\n", particles.size(), t_diff);
		}
		else if (displayCnt == 50)
		{
			displayCnt++;
			printf("Particles: %lu \t Computation time: %.2f\n", particles.size(), timeSum / 50.);
		}
	}

	if (useLighting)
	{
		glDisable(GL_LIGHT0);
		glDisable(GL_LIGHTING);
	}

#ifdef DEMO_Z_FIGHTING
	if (DepthFightingOn != 0)
	{
		glPushMatrix();
		glRotatef(90.f, 0.f, 1.f, 0.f);
		glCallList(ParticleList);
		glPopMatrix();
	}
#endif

	// draw some gratuitous text that just rotates on top of the scene:
	// i commented out the actual text-drawing calls -- put them back in if you have a use for them
	// a good use for thefirst one might be to have your name on the screen
	// a good use for the second one might be to have vertex numbers on the screen alongside each vertex

	glDisable(GL_DEPTH_TEST);
	glColor3f(0.f, 1.f, 1.f);
	// DoRasterString( 0.f, 1.f, 0.f, (char *)"Text That Moves" );

	// draw some gratuitous text that is fixed on the screen:
	//
	// the projection matrix is reset to define a scene whose
	// world coordinate system goes from 0-100 in each axis
	//
	// this is called "percent units", and is just a convenience
	//
	// the modelview matrix is reset to identity as we don't
	// want to transform these coordinates

	glDisable(GL_DEPTH_TEST);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.f, 100.f, 0.f, 100.f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glColor3f(1.f, 1.f, 1.f);
	// string to be displayed on screen
	std::string textToDisplay1 = std::to_string(particles.size()) + " Particles";
	std::string textToDisplay2 = "Rest density: " + std::to_string((int)rest_density);
	char* textCharArray1 = &textToDisplay1[0u];
	char* textCharArray2 = &textToDisplay2[0u];
	DoRasterString(5.f, 7.f, 0.f, textCharArray1);
	DoRasterString(5.f, 2.5f, 0.f, textCharArray2);

	// swap the double-buffered framebuffers:

	glutSwapBuffers();

	// be sure the graphics buffer has been sent:
	// note: be sure to use glFlush( ) here, not glFinish( ) !

	glFlush();
}

void DoAxesMenu(int id)
{
	AxesOn = id;

	glutSetWindow(MainWindow);
	glutPostRedisplay();
}

void DoColorMenu(int id)
{
	NowColor = id - RED;

	glutSetWindow(MainWindow);
	glutPostRedisplay();
}

void DoDebugMenu(int id)
{
	DebugOn = id;

	glutSetWindow(MainWindow);
	glutPostRedisplay();
}

void DoDepthBufferMenu(int id)
{
	DepthBufferOn = id;

	glutSetWindow(MainWindow);
	glutPostRedisplay();
}

void DoDepthFightingMenu(int id)
{
	DepthFightingOn = id;

	glutSetWindow(MainWindow);
	glutPostRedisplay();
}

void DoDepthMenu(int id)
{
	DepthCueOn = id;

	glutSetWindow(MainWindow);
	glutPostRedisplay();
}

// main menu callback:

void DoMainMenu(int id)
{
	switch (id)
	{
	case RESET:
		Reset();
		break;

	case QUIT:
		// gracefully close out the graphics:
		// gracefully close the graphics window:
		// gracefully exit the program:
		glutSetWindow(MainWindow);
		glFinish();
		glutDestroyWindow(MainWindow);
		exit(0);
		break;

	default:
		fprintf(stderr, "Don't know what to do with Main Menu ID %d\n", id);
	}

	glutSetWindow(MainWindow);
	glutPostRedisplay();
}

void DoProjectMenu(int id)
{
	NowProjection = id;

	glutSetWindow(MainWindow);
	glutPostRedisplay();
}

// use glut to display a string of characters using a raster font:

void DoRasterString(float x, float y, float z, char* s)
{
	glRasterPos3f((GLfloat)x, (GLfloat)y, (GLfloat)z);

	char c; // one character to print
	for (; (c = *s) != '\0'; s++)
	{
		glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, c);
	}
}

// use glut to display a string of characters using a stroke font:

void DoStrokeString(float x, float y, float z, float ht, char* s)
{
	glPushMatrix();
	glTranslatef((GLfloat)x, (GLfloat)y, (GLfloat)z);
	float sf = ht / (119.05f + 33.33f);
	glScalef((GLfloat)sf, (GLfloat)sf, (GLfloat)sf);
	char c; // one character to print
	for (; (c = *s) != '\0'; s++)
	{
		glutStrokeCharacter(GLUT_STROKE_ROMAN, c);
	}
	glPopMatrix();
}

// return the number of seconds since the start of the program:

float ElapsedSeconds()
{
	// get # of milliseconds since the start of the program:

	int ms = glutGet(GLUT_ELAPSED_TIME);

	// convert it to seconds:

	return (float)ms / 1000.f;
}

// initialize the glui window:

void InitMenus()
{
	if (DebugOn != 0)
		fprintf(stderr, "Starting InitMenus.\n");

	glutSetWindow(MainWindow);

	int numColors = sizeof(Colors) / (3 * sizeof(float));
	int colormenu = glutCreateMenu(DoColorMenu);
	for (int i = 0; i < numColors; i++)
	{
		glutAddMenuEntry(ColorNames[i], i);
	}

	int axesmenu = glutCreateMenu(DoAxesMenu);
	glutAddMenuEntry("Off", 0);
	glutAddMenuEntry("On", 1);

	int depthcuemenu = glutCreateMenu(DoDepthMenu);
	glutAddMenuEntry("Off", 0);
	glutAddMenuEntry("On", 1);

	int depthbuffermenu = glutCreateMenu(DoDepthBufferMenu);
	glutAddMenuEntry("Off", 0);
	glutAddMenuEntry("On", 1);

	int depthfightingmenu = glutCreateMenu(DoDepthFightingMenu);
	glutAddMenuEntry("Off", 0);
	glutAddMenuEntry("On", 1);

	int debugmenu = glutCreateMenu(DoDebugMenu);
	glutAddMenuEntry("Off", 0);
	glutAddMenuEntry("On", 1);

	int projmenu = glutCreateMenu(DoProjectMenu);
	glutAddMenuEntry("Orthographic", ORTHO);
	glutAddMenuEntry("Perspective", PERSP);

	int mainmenu = glutCreateMenu(DoMainMenu);
	glutAddSubMenu("Axes", axesmenu);
	glutAddSubMenu("Axis Colors", colormenu);

#ifdef DEMO_DEPTH_BUFFER
	glutAddSubMenu("Depth Buffer", depthbuffermenu);
#endif

#ifdef DEMO_Z_FIGHTING
	glutAddSubMenu("Depth Fighting", depthfightingmenu);
#endif

	glutAddSubMenu("Depth Cue", depthcuemenu);
	glutAddSubMenu("Projection", projmenu);
	glutAddMenuEntry("Reset", RESET);
	glutAddSubMenu("Debug", debugmenu);
	glutAddMenuEntry("Quit", QUIT);

	// attach the pop-up menu to the right mouse button:

	glutAttachMenu(GLUT_RIGHT_BUTTON);
}

// initialize the glut and OpenGL libraries:
//	also setup callback functions

void InitGraphics()
{
	if (DebugOn != 0)
		fprintf(stderr, "Starting InitGraphics.\n");

	// request the display modes:
	// ask for red-green-blue-alpha color, double-buffering, and z-buffering:

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);

	// set the initial window configuration:

	glutInitWindowPosition(0, 0);
	glutInitWindowSize(INIT_WINDOW_SIZE, INIT_WINDOW_SIZE);

	// open the window and set its title:

	MainWindow = glutCreateWindow(WINDOWTITLE);
	glutSetWindowTitle(WINDOWTITLE);

	// set the framebuffer clear values:

	glClearColor(BACKCOLOR[0], BACKCOLOR[1], BACKCOLOR[2], BACKCOLOR[3]);

	// setup the callback functions:
	// DisplayFunc -- redraw the window
	// ReshapeFunc -- handle the user resizing the window
	// KeyboardFunc -- handle a keyboard input
	// MouseFunc -- handle the mouse button going down or up
	// MotionFunc -- handle the mouse moving with a button down
	// PassiveMotionFunc -- handle the mouse moving with a button up
	// VisibilityFunc -- handle a change in window visibility
	// EntryFunc	-- handle the cursor entering or leaving the window
	// SpecialFunc -- handle special keys on the keyboard
	// SpaceballMotionFunc -- handle spaceball translation
	// SpaceballRotateFunc -- handle spaceball rotation
	// SpaceballButtonFunc -- handle spaceball button hits
	// ButtonBoxFunc -- handle button box hits
	// DialsFunc -- handle dial rotations
	// TabletMotionFunc -- handle digitizing tablet motion
	// TabletButtonFunc -- handle digitizing tablet button hits
	// MenuStateFunc -- declare when a pop-up menu is in use
	// TimerFunc -- trigger something to happen a certain time from now
	// IdleFunc -- what to do when nothing else is going on

	glutSetWindow(MainWindow);
	glutDisplayFunc(Display);
	glutReshapeFunc(Resize);
	glutKeyboardFunc(Keyboard);
	glutMouseFunc(MouseButton);
	glutMotionFunc(MouseMotion);
	glutPassiveMotionFunc(MouseMotion);
	// glutPassiveMotionFunc( NULL );
	glutVisibilityFunc(Visibility);
	glutEntryFunc(NULL);
	glutSpecialFunc(NULL);
	glutSpaceballMotionFunc(NULL);
	glutSpaceballRotateFunc(NULL);
	glutSpaceballButtonFunc(NULL);
	glutButtonBoxFunc(NULL);
	glutDialsFunc(NULL);
	glutTabletMotionFunc(NULL);
	glutTabletButtonFunc(NULL);
	glutMenuStateFunc(NULL);
	glutTimerFunc(-1, NULL, 0);

	// setup glut to call Animate( ) every time it has
	// 	nothing it needs to respond to (which is most of the time)
	// we don't need to do this for this program, and really should set the argument to NULL
	// but, this sets us up nicely for doing animation

	glutIdleFunc(Animate);

	// init the glew package (a window must be open to do this):

#ifdef WIN32
	GLenum err = glewInit();
	if (err != GLEW_OK)
	{
		fprintf(stderr, "glewInit Error\n");
	}
	else
		fprintf(stderr, "GLEW initialized OK\n");
	fprintf(stderr, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
#endif

	// all other setups go here, such as GLSLProgram and KeyTime setups:
}

// initialize the display lists that will not change:
// (a display list is a way to store opengl commands in
//  memory so that they can be played back efficiently at a later time
//  with a call to glCallList( )

void InitLists()
{
	if (DebugOn != 0)
		fprintf(stderr, "Starting InitLists.\n");

	glutSetWindow(MainWindow);

	// create the object:

	ParticleList = glGenLists(1);
	glNewList(ParticleList, GL_COMPILE);
	OsuSphere(0.03, 8, 8);
	glEndList();

#define YGRID	-0.07f

#define XSIDE1	SIM_W*2			// length of the x side of the grid
#define X01      (-XSIDE1/2.)		// where one side starts
#define NX1	50			// how many points in x
#define DX1	( XSIDE1/(float)NX1 )	// change in x between the points

#define ZSIDE1	SIM_W*2			// length of the z side of the grid
#define Z01      (-ZSIDE1/2.)		// where one side starts
#define NZ1	50			// how many points in z
#define DZ1	( ZSIDE1/(float)NZ1 )	// change in z between the points

	// int NZ = 100, NX = 100;
	// float xside = SIM_W*5, zside = SIM_W*5;
	// float X0 = xside/2.f, Z0 = zside/2.f;
	// float DX = xside/(float)NX, DZ = zside/(float)NZ;
	// float YGRID = 0.f;
	GridDL1 = glGenLists(1);
	glNewList(GridDL1, GL_COMPILE);
	SetMaterial(1.f, 1.f, .6f, 10.f);
	glNormal3f(0., 1., 0.);
	for (int i = 0; i < NZ1; i++)
	{
		glBegin(GL_QUAD_STRIP);
		for (int j = 0; j < NX1; j++)
		{
			glVertex3f(X01 + DX1 * (float)j, YGRID, Z01 + DZ1 * (float)(i + 0));
			glVertex3f(X01 + DX1 * (float)j, YGRID, Z01 + DZ1 * (float)(i + 1));
		}
		glEnd();
	}
	glEndList();

#define XSIDE2	SIM_W*6			// length of the x side of the grid
#define X02      (-XSIDE2/2.)		// where one side starts
#define NX2	150			// how many points in x
#define DX2	( XSIDE2/(float)NX2 )	// change in x between the points

#define ZSIDE2	SIM_W*2			// length of the z side of the grid
#define Z02      (-ZSIDE2/2.)		// where one side starts
#define NZ2	50			// how many points in z
#define DZ2	( ZSIDE2/(float)NZ2 )	// change in z between the points

	// int NZ = 100, NX = 100;
	// float xside = SIM_W*5, zside = SIM_W*5;
	// float X0 = xside/2.f, Z0 = zside/2.f;
	// float DX = xside/(float)NX, DZ = zside/(float)NZ;
	// float YGRID = 0.f;
	GridDL2 = glGenLists(1);
	glNewList(GridDL2, GL_COMPILE);
	SetMaterial(0.3f, .8f, 1.f, 10.f);
	glNormal3f(0., 1., 0.);
	for (int i = 0; i < NZ2; i++)
	{
		glBegin(GL_QUAD_STRIP);
		for (int j = 0; j < NX2; j++)
		{
			glVertex3f(X02 + DX2 * (float)j, YGRID, Z02 + DZ2 * (float)(i + 0));
			glVertex3f(X02 + DX2 * (float)j, YGRID, Z02 + DZ2 * (float)(i + 1));
		}
		glEnd();
	}
	glEndList();

	// create the axes:

	AxesList = glGenLists(1);
	glNewList(AxesList, GL_COMPILE);
	glLineWidth(AXES_WIDTH);
	Axes(1.5);
	glLineWidth(1.);
	glEndList();
}

// the keyboard callback:

void Keyboard(unsigned char c, int x, int y)
{
	if (DebugOn != 0)
		fprintf(stderr, "Keyboard: '%c' (0x%0x)\n", c, c);

	switch (c)
	{
	case 'o':
	case 'O':
		// NowProjection = ORTHO;
		useOpening = !useOpening;
		break;

	case 'p':
	case 'P':
		NowProjection = PERSP;
		break;

	case 'q':
	case 'Q':
	case ESCAPE:
		DoMainMenu(QUIT); // will not return here
		break;			  // happy compiler

	case 's':
	case 'S':
		doSimulation = !doSimulation;
		break;

	case ' ':
		addMoreParticles(500);
		break;

	case 'l':
	case 'L':
		useLighting = !useLighting;
		break;

	case 'g':
	case 'G':
		useGravity = !useGravity;
		break;

	case '1':
		rest_density = 1.;
		break;

	case '2':
		rest_density = 2.;
		break;

	case '3':
		rest_density = 3.;
		break;

	case '4':
		rest_density = 4.;
		break;

	case '5':
		rest_density = 5.;
		break;

	case '6':
		rest_density = 6.;
		break;

	case '7':
		rest_density = 7.;
		break;

	case '8':
		rest_density = 8.;
		break;

	case '9':
		rest_density = 9.;
		break;

	case '0':
		rest_density = 10.;
		break;

		// used for gathering graph data
	case 'a':
		// particles.clear();
		addMoreParticles(200);
		displayCnt = 0;
		timeSum = 0;
		break;

	case 'c':
		useColorVisual = !useColorVisual;
		break;

	case 'e':
		externalForce = !externalForce;
		break;

	case 'r':
		shrinkWorld = !shrinkWorld;
		break;

	default:
		fprintf(stderr, "Don't know what to do with keyboard hit: '%c' (0x%0x)\n", c, c);
	}

	// force a call to Display( ):

	glutSetWindow(MainWindow);
	glutPostRedisplay();
}

// called when the mouse button transitions down or up:

void MouseButton(int button, int state, int x, int y)
{
	int b = 0; // LEFT, MIDDLE, or RIGHT

	if (DebugOn != 0)
		fprintf(stderr, "MouseButton: %d, %d, %d, %d\n", button, state, x, y);

	// get the proper button bit mask:

	switch (button)
	{
	case GLUT_LEFT_BUTTON:
		b = LEFT;
		break;

	case GLUT_MIDDLE_BUTTON:
		b = MIDDLE;
		break;

	case GLUT_RIGHT_BUTTON:
		b = RIGHT;
		break;

	case SCROLL_WHEEL_UP:
		Scale += SCLFACT * SCROLL_WHEEL_CLICK_FACTOR;
		// keep object from turning inside-out or disappearing:
		if (Scale < MINSCALE)
			Scale = MINSCALE;
		break;

	case SCROLL_WHEEL_DOWN:
		Scale -= SCLFACT * SCROLL_WHEEL_CLICK_FACTOR;
		// keep object from turning inside-out or disappearing:
		if (Scale < MINSCALE)
			Scale = MINSCALE;
		break;

	default:
		b = 0;
		fprintf(stderr, "Unknown mouse button: %d\n", button);
	}

	// button down sets the bit, up clears the bit:

	if (state == GLUT_DOWN)
	{
		Xmouse = x;
		Ymouse = y;
		ActiveButton |= b; // set the proper bit
	}
	else
	{
		ActiveButton &= ~b; // clear the proper bit
	}

	glutSetWindow(MainWindow);
	glutPostRedisplay();
}

// called when the mouse moves while a button is down:

void MouseMotion(int x, int y)
{
	int dx = x - Xmouse; // change in mouse coords
	int dy = y - Ymouse;

	if ((ActiveButton & LEFT) != 0)
	{
		Xrot += (ANGFACT * dy);
		Yrot += (ANGFACT * dx);
	}

	if ((ActiveButton & MIDDLE) != 0)
	{
		Scale += SCLFACT * (float)(dx - dy);

		// keep object from turning inside-out or disappearing:

		if (Scale < MINSCALE)
			Scale = MINSCALE;
	}

	Xmouse = x; // new current position
	Ymouse = y;

	glutSetWindow(MainWindow);
	glutPostRedisplay();
}

// reset the transformations and the colors:
// this only sets the global variables --
// the glut main loop is responsible for redrawing the scene

void Reset()
{
	ActiveButton = 0;
	AxesOn = 1;
	DebugOn = 0;
	DepthBufferOn = 1;
	DepthFightingOn = 0;
	DepthCueOn = 0;
	Scale = 1.0;
	ShadowsOn = 0;
	NowColor = YELLOW;
	NowProjection = PERSP;
	Xrot = Yrot = 0.;

	displayCnt = 0;
	timeSum = 0;
	particles.clear();
	initParticles(N);
	doSimulation = false;
	useGravity = true;
	useColorVisual = false;
	externalForce = false;
	shrinkWorld = true;
	useLighting = false;
	useOpening = false;
}

// called when user resizes the window:

void Resize(int width, int height)
{
	// don't really need to do anything since window size is
	// checked each time in Display( ):

	glutSetWindow(MainWindow);
	glutPostRedisplay();
}

// handle a change to the window's visibility:

void Visibility(int state)
{
	if (DebugOn != 0)
		fprintf(stderr, "Visibility: %d\n", state);

	if (state == GLUT_VISIBLE)
	{
		glutSetWindow(MainWindow);
		glutPostRedisplay();
	}
	else
	{
		// could optimize by keeping track of the fact
		// that the window is not visible and avoid
		// animating or redrawing it ...
	}
}

///////////////////////////////////////   HANDY UTILITIES:  //////////////////////////

// the stroke characters 'X' 'Y' 'Z' :

static float xx[] = { 0.f, 1.f, 0.f, 1.f };

static float xy[] = { -.5f, .5f, .5f, -.5f };

static int xorder[] = { 1, 2, -3, 4 };

static float yx[] = { 0.f, 0.f, -.5f, .5f };

static float yy[] = { 0.f, .6f, 1.f, 1.f };

static int yorder[] = { 1, 2, 3, -2, 4 };

static float zx[] = { 1.f, 0.f, 1.f, 0.f, .25f, .75f };

static float zy[] = { .5f, .5f, -.5f, -.5f, 0.f, 0.f };

static int zorder[] = { 1, 2, 3, 4, -5, 6 };

// fraction of the length to use as height of the characters:
const float LENFRAC = 0.10f;

// fraction of length to use as start location of the characters:
const float BASEFRAC = 1.10f;

//	Draw a set of 3D axes:
//	(length is the axis length in world coordinates)

void Axes(float length)
{
	glBegin(GL_LINE_STRIP);
	glVertex3f(length, 0., 0.);
	glVertex3f(0., 0., 0.);
	glVertex3f(0., length, 0.);
	glEnd();
	glBegin(GL_LINE_STRIP);
	glVertex3f(0., 0., 0.);
	glVertex3f(0., 0., length);
	glEnd();

	float fact = LENFRAC * length;
	float base = BASEFRAC * length;

	glBegin(GL_LINE_STRIP);
	for (int i = 0; i < 4; i++)
	{
		int j = xorder[i];
		if (j < 0)
		{

			glEnd();
			glBegin(GL_LINE_STRIP);
			j = -j;
		}
		j--;
		glVertex3f(base + fact * xx[j], fact * xy[j], 0.0);
	}
	glEnd();

	glBegin(GL_LINE_STRIP);
	for (int i = 0; i < 5; i++)
	{
		int j = yorder[i];
		if (j < 0)
		{

			glEnd();
			glBegin(GL_LINE_STRIP);
			j = -j;
		}
		j--;
		glVertex3f(fact * yx[j], base + fact * yy[j], 0.0);
	}
	glEnd();

	glBegin(GL_LINE_STRIP);
	for (int i = 0; i < 6; i++)
	{
		int j = zorder[i];
		if (j < 0)
		{

			glEnd();
			glBegin(GL_LINE_STRIP);
			j = -j;
		}
		j--;
		glVertex3f(0.0, fact * zy[j], base + fact * zx[j]);
	}
	glEnd();
}

// function to convert HSV to RGB
// 0.  <=  s, v, r, g, b  <=  1.
// 0.  <= h  <=  360.
// when this returns, call:
//		glColor3fv( rgb );

void HsvRgb(float hsv[3], float rgb[3])
{
	// guarantee valid input:

	float h = hsv[0] / 60.f;
	while (h >= 6.)
		h -= 6.;
	while (h < 0.)
		h += 6.;

	float s = hsv[1];
	if (s < 0.)
		s = 0.;
	if (s > 1.)
		s = 1.;

	float v = hsv[2];
	if (v < 0.)
		v = 0.;
	if (v > 1.)
		v = 1.;

	// if sat==0, then is a gray:

	if (s == 0.0)
	{
		rgb[0] = rgb[1] = rgb[2] = v;
		return;
	}

	// get an rgb from the hue itself:

	float i = (float)floor(h);
	float f = h - i;
	float p = v * (1.f - s);
	float q = v * (1.f - s * f);
	float t = v * (1.f - (s * (1.f - f)));

	float r = 0., g = 0., b = 0.; // red, green, blue
	switch ((int)i)
	{
	case 0:
		r = v;
		g = t;
		b = p;
		break;

	case 1:
		r = q;
		g = v;
		b = p;
		break;

	case 2:
		r = p;
		g = v;
		b = t;
		break;

	case 3:
		r = p;
		g = q;
		b = v;
		break;

	case 4:
		r = t;
		g = p;
		b = v;
		break;

	case 5:
		r = v;
		g = p;
		b = q;
		break;
	}

	rgb[0] = r;
	rgb[1] = g;
	rgb[2] = b;
}

void Cross(float v1[3], float v2[3], float vout[3])
{
	float tmp[3];
	tmp[0] = v1[1] * v2[2] - v2[1] * v1[2];
	tmp[1] = v2[0] * v1[2] - v1[0] * v2[2];
	tmp[2] = v1[0] * v2[1] - v2[0] * v1[1];
	vout[0] = tmp[0];
	vout[1] = tmp[1];
	vout[2] = tmp[2];
}

float Dot(float v1[3], float v2[3])
{
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

float Unit(float vin[3], float vout[3])
{
	float dist = vin[0] * vin[0] + vin[1] * vin[1] + vin[2] * vin[2];
	if (dist > 0.0)
	{
		dist = sqrtf(dist);
		vout[0] = vin[0] / dist;
		vout[1] = vin[1] / dist;
		vout[2] = vin[2] / dist;
	}
	else
	{
		vout[0] = vin[0];
		vout[1] = vin[1];
		vout[2] = vin[2];
	}
	return dist;
}

float Unit(float v[3])
{
	float dist = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
	if (dist > 0.0)
	{
		dist = sqrtf(dist);
		v[0] /= dist;
		v[1] /= dist;
		v[2] /= dist;
	}
	return dist;
}