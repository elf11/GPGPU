#ifndef _CORE_COMP_
#define _CORE_COMP_
#include <vector>
#include<GL/glew.h>
#include<GL/GL.H>
#include<GL/GLU.H>
#include<GL/GLAUX.H>
#include<GL/glut.h>
#include<math.h>
#include<algorithm>

#define PI 3.14159265
#define MAX(a, b) ((a>b)?a:b)
#define MIN(a, b) ((a<b)?a:b)

/**
* \struct Vector3 Structura principala care reprezinta un punct
* in spatiul 3d
*/
struct Vector3{
  float x, y, z;
  Vector3(float x, float y, float z);
  Vector3();
  ~Vector3();

  /**
  * \return Va insuma elementele vectorilor.
  * (a1, b1, c1) + (a2, b2, c2) = (a1+a2, b1+b2, c1+c2)
  */
  Vector3 operator+(Vector3 &x);

  /**
  \return The answer will be the difference of the two
  * vectors
  * (a1, b1, c1) - (a2, b2, c2) = (a1-a2, b1-b2, c1-c2)
  */
  Vector3 operator-(Vector3 &x);

  /**
  * \brief Imparte fiecare element al vetorului la \param m
  */
  Vector3 operator/(float &m);

  /**
  * \brief multiplica fiecare element al vectorului cu m
  */
  Vector3 operator*(float &m);

  /**
  * \brief Afla distanta intre vectorul implicit si
  * \param a
  */
  float getDist(Vector3 &a);

  /**
  * \brief Am convenit ca vectorul cu primul element -1 sa fie invalid
  * metoda returneaza true daca elementul e valid false altfel
  */
  virtual bool isValid();

  /**
  * \brief apeleaza functia glTranslatef cu parametrii 
  * vectorului <b>point</b>
  * \param point
  */
  static void translate(Vector3 point);

  /**
  * \brief Apeleaza glRotatef cu quaternion-ul
  * \param point si \param angle
  */
  static void rotate(Vector3 point, float angle);

  /**
  * \brief apeleaza functia glScalef cu parametrii
  * \param point
  * \attention Nu e testata!
  */
  static void scale(Vector3 point);

  /**
  * \brief Face miscare de translatie inversa pentru 
  * \param point
  */
  static void translateBack(Vector3 point);

  /**
  * \brief Face rotatia inversa pentru quaternionul 
  * \param point si \param angle
  */
  static void rotateBack(Vector3 point, float angle);

  /**
  * \brief Face scalarea inversa pentru \param point
  */
  static void scaleBack(Vector3 point);

  /**
  * \brief calculeaza distanta intre 2 puncte in spatiu
  */
  static float getDist(Vector3 &a, Vector3 &b);

  /**
  * \brief Returneaza vectorul invalid conform "intelegerii"
  * \attention Trebuie avut grija la intervalul vectorilor valizi
  * sa nu se suprapuna peste vectorul invalid
  */
  static Vector3 invalidPoint();

  /**
  * \brief Spne daca toate coordonatele vectorului curent sunt intre
  * coordonatele celor 2 vectori
  */
  bool isInside(Vector3 &a, Vector3 &b);

  /**
  * \brief utilizata pentru Debugging, afiseaza coordonatele punctului
  * curent.
  * \attention Desenarea pe ecran nu este testata
  */
  void print();
};

/**
* \struct Point2D Defineste un punct pe axele ox, oy
*/
struct Point2D{
  float x, y;
  Point2D(float x, float y);
  Point2D();
  ~Point2D();
};

/**
* \struct Vector3Int Este un caz mai particular, este folosit pentru indexarea
* fetelor, ar putea fi inclus in clasa Face
*/
struct Vector3Int{
  int x, y, z;
  Vector3Int(int x, int y, int z);
  Vector3Int();
  ~Vector3Int();
};

/**
* \struct Face Defineste o fata triunghiulara
* \param v - Indexele vertecsilor dintr-un vector in care sunt pastrati vertecsii
* \param uv - indecsi spre vectorul cu coordonatele texturii pentru
* fiecare vertex
* \param n - indecsi spre vectorul cu normalele
* \details Initial Desenarea se facea de pe CPU, foloseam aceasta clasa
* ca sa consum mai putina memorie, o data cu trecerea la VBO aceasta 
* structura a devenit deprecated pentru scopul pentru care a fost conceputa, insa poate fi utila in alte situatii
*/
struct Face{
  Vector3Int v, uv, n;
  Face();
  Face(Vector3Int v, Vector3Int uv, Vector3Int n);
  ~Face();
};

/**
* \struct TexelRGBA Este folosita exclusiv pentru citirea imaginilor TGA,
* poate fi utila in alte situatii
*/
struct TexelRGBA{
  unsigned char r, g, b, a;
};

/**
* \struct TexelRGB Este folosita exclusiv pentru citirea imaginilor TGA,
* poate fi utila in alte situatii
*/
struct TexelRGB{
  unsigned char r, g, b;
};

/**
* \class BoundingBox Clasa generala pentru bounding-box-uri
*/
class BoundingBox{
public:
  /**
  * \brief Se poate dezvolta aplicatia astfel incat un bounding-box sa
  * fie de oricate forme.
  */
  enum{SPHERE, PARALLELEPIPED};
  /**
  * \param tipul frame-ului este folosit pentru a cunoaste clasa derivata
  * la care se face cast
  */
  int type;
  BoundingBox();
  ~BoundingBox();
  virtual bool colide(BoundingBox *b);

  /**
  * \return Returneaza tipul bounding box-ului
  */
  inline int getType() const {return type;}
};

/**
* \class AAParallelepiped Clasa pentru Bounding box de tip paralelipiped
*/
class AAParallelepiped : public BoundingBox{
  public:
  /**
  * \param minPoint si \param maxPoint definesc un paralelipiped
  */
  Vector3 minPoint, maxPoint;
  AAParallelepiped();
  AAParallelepiped(Vector3 &minPoint, Vector3 &maxPoint);
  AAParallelepiped(float minPointx, 
    float minPointy, 
    float minPointz, 
    float maxPointx,
    float maxPointy,
    float maxPointz);
  ~AAParallelepiped();
  
  /**
  * \return Returneaza true daca obiectul curent se ciocneste cu bounding-
  * box-ul \param b
  */
  virtual bool colide(BoundingBox *b);
};

class Sphere:public BoundingBox{
public:
  float radius;
  Sphere();
  Sphere(float radius);
  ~Sphere();
};


/**
* \class object Clasa care defineste un mesh
*/
struct object{
  char name[40];
  /**
  * \param vertices \param normals \param textMap \param face
  * Vectori temporari care sunt goi in cea mai mare parte a timpului
  * Ar putea fi eliminati, dar cu grija, sunt mostenire de cand
  * se desenau mesh-urile din procesor
  */
  std::vector<Vector3> vertices;
  std::vector<Vector3> normals;
  std::vector<Point2D> texMap;
  std::vector<Face> face;

  GLuint textureId;
  GLuint VBOID, IBOID;
  std::vector<BoundingBox*> bounding_box;
  Vector3 location;
  /**
  * \param duplicates Reprezinta replicarile aceluiasi mesh la diferite
  * coordonate
  */
  std::vector<Vector3> duplicates;
  /**
  * \param angle si \param rotation formeaza un quaternion pentru rotatie
  */
  float angle;
  Vector3 rotation;
  Vector3 scaling;
  object();
  ~object();

  /**
  * \brief Citeste un bounding box dintr-un fisier.A se vedea formatul
  * fisierelor
  */
  void setBoundingBox(const char* filename);
};
#endif