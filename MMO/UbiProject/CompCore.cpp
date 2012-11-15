#include "CompCore.h"

Vector3::Vector3(float x, float y, float z)
{
  this->x = x;
  this->y = y;
  this->z = z;
}

Vector3::Vector3()
{
  this->x = 0;
  this->y = 0;
  this->z = 0;
}

Vector3::~Vector3()
{
}

Vector3 Vector3::operator+(Vector3 &x)
{
  Vector3 rez;
  rez.x = this->x + x.x;
  rez.y = this->y + x.y;
  rez.z = this->z + x.z;
  return rez;
}

Vector3 Vector3::operator-(Vector3 &x)
{
  Vector3 rez;
  rez.x = this->x - x.x;
  rez.y = this->y - x.y;
  rez.z = this->z - x.z;
  return rez;
}

Vector3 Vector3::operator/(float &m)
{
  Vector3 rez;
  rez.x = this->x / m;
  rez.y = this->y / m;
  rez.z = this->z / m;
  return rez;
}

Vector3 Vector3::operator*(float &m)
{
  Vector3 rez;
  rez.x = this->x * m;
  rez.y = this->y * m;
  rez.z = this->z * m;
  return rez;
}
void Vector3::translate(Vector3 point)
{
  glTranslatef(point.x, point.y, point.z);
}

void Vector3::rotate(Vector3 point, float angle)
{
    glRotatef(angle, point.x, point.y, point.z);
}

void Vector3::scale(Vector3 point)
{
  glScalef(point.x, point.y, point.z);
}

void Vector3::translateBack(Vector3 point)
{
  glTranslatef(-point.x, -point.y, -point.z);
}

void Vector3::rotateBack(Vector3 point, float angle)
{
    glRotatef(-angle, point.x, point.y, point.z);
}

void Vector3::scaleBack(Vector3 point)
{
  //glScalef(-point.x, -point.y, -point.z);
}

float Vector3::getDist(Vector3 &a)
{
  return sqrt(pow((a.x - this->x), 2.0f)+pow((a.y - this->y), 2.0f)+pow(a.z - this->z, 2.0f));
}

float Vector3::getDist(Vector3 &a, Vector3 &b)
{
return sqrt(pow((a.x - b.x), 2.0f)+pow((a.y - b.y), 2.0f)+pow(a.z - b.z, 2.0f));
}

Vector3 Vector3::invalidPoint()
{
  Vector3 x;
  x.x = -1;
  x.y = 0;
  x.z = 0;
  return x;
}

bool Vector3::isInside(Vector3 &a, Vector3 &b)
{
 if(MIN(MIN(a.x, b.x), this->x) >= this->x || MAX(MAX(a.x, b.x), this->x) <= this->x)
   return false;
 if(MIN(MIN(a.y, b.y), this->y) >= this->y || MAX(MAX(a.y, b.y), this->y) <= this->y)
   return false;
 if(MIN(MIN(a.z, b.z), this->z) >= this->z || MAX(MAX(a.z, b.z), this->z) <= this->z)
   return false;
 return true;
}

bool Vector3::isValid()
{
  if(x == -1)
    return true;
  return false;
}

void Vector3::print()
{
  glPointSize(1.0f);
  glBegin(GL_POINTS);
    glColor3f(1, 0, 0);
    glVertex3f(-x, y, -z-100);
  glEnd();
  printf("%f %f %f\n", x, y, z);
}

Vector3Int::Vector3Int()
{
}


Vector3Int::Vector3Int(int x, int y, int z)
{
  this->x = x;
  this->y = y;
  this->z = z;
}

Vector3Int::~Vector3Int()
{
}

Point2D::Point2D()
{
  x = y = 0.0f;
}

Point2D::Point2D(float x, float y)
{
  this->x = x;
  this->y = y;
}

Point2D::~Point2D()
{
}

Face::Face()
{
}

Face::Face(Vector3Int v, Vector3Int uv, Vector3Int n)
{
  this->v = v;
  this->uv = uv;
  this->n = n;
}

Face::~Face()
{
}

BoundingBox::BoundingBox()
{
}

BoundingBox::~BoundingBox()
{
}

bool BoundingBox::colide(BoundingBox *b)
{
  if(this->getType() == BoundingBox::PARALLELEPIPED){
    AAParallelepiped *p = reinterpret_cast<AAParallelepiped*>(this);
    return p->colide(b);
  }
  return false;
}

AAParallelepiped::AAParallelepiped()
{
  Vector3 a, b;
  minPoint = a;
  maxPoint = b;
}

AAParallelepiped::AAParallelepiped(Vector3 &minPoint, Vector3 &maxPoint)
{
  this->minPoint = minPoint;
  this->maxPoint = maxPoint;
}

AAParallelepiped::AAParallelepiped(float minPointx, 
    float minPointy, 
    float minPointz, 
    float maxPointx,
    float maxPointy,
    float maxPointz)
{
   this->minPoint.x = minPointx;
   this->minPoint.y = minPointy;
   this->minPoint.z = minPointz;
   this->maxPoint.x = maxPointx;
   this->maxPoint.y = maxPointy;
   this->maxPoint.z = maxPointz;
}
AAParallelepiped::~AAParallelepiped()
{
}

bool AAParallelepiped::colide(BoundingBox *b)
{

  if(b->getType() == PARALLELEPIPED){
    AAParallelepiped *p = reinterpret_cast<AAParallelepiped*>(b);
    return (this->minPoint.isInside(p->minPoint, p->maxPoint) || 
      this->maxPoint.isInside(p->minPoint, p->maxPoint));
  }
  return false;
}

Sphere::Sphere()
{
  this->radius = radius;
}

Sphere::Sphere(float radius)
{
  this->radius = radius;
}

Sphere::~Sphere()
{
}

object::object():bounding_box(NULL), VBOID(0), IBOID(0)
{
  strcpy(name, "unidentified");
}

object::~object()
{/*
  VBOID;
  if(VBOID > 0){
    printf("::: %s\n", name);
    glDeleteBuffers(1, &VBOID);
    VBOID = 0;
  }
  */
  /*
  if(IBOID > 0)
    glDeleteBuffers(1, &IBOID);
    */
}

void object::setBoundingBox(const char* filename)
{
  FILE *f = fopen(filename, "r");
    if(f != NULL){
      int n;
      fscanf(f, "%d", &n);
      for(int i = 0; i < n; i++){
        int type;
        fscanf(f, "%d", &type);
        if(type == 0){
          float minX, minY, minZ, maxX, maxY, maxZ;
          fscanf(f, "%f %f %f %f %f %f", &minX, &minY, &minZ, &maxX, &maxY, &maxZ);
          AAParallelepiped *p = 
            new AAParallelepiped(minX, minY, minZ, maxX, maxY, maxZ);
          p->type = BoundingBox::PARALLELEPIPED;
          bounding_box.push_back(p);
        }
      }
      fclose(f);
    }
}
