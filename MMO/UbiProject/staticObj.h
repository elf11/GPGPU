#ifndef _STATIC_OBJ_LIB_
#define _STATIC_OBJ_LIB_

#pragma comment(lib, "glew32.lib")

#include <string.h>
#include <math.h>
#include <vector>
#include <fstream>
#include <iostream>
#include "CompCore.h"
#include "string.h"
#include "TGAImage.h"
#include "physicEngine.h"
#include "camera.h"


#define BUFF_SIZE 256
/**
* \class StaticObj Reprezinta toate obiectele statice din scena 
* sau care se comporta periodic
*/
class StaticObj{
public:
  std::vector<object> components;

  /**
  * \param objFile este numele fisierului de configuare 
  * A se vedea formatul fisierelor
  */
  explicit StaticObj(const char* objFile);
  ~StaticObj();

  /**
  * \brief citeste in \param comp, obiectul de tip.obj cde la adresa
  * \param filename
  */
  static void readObj(char* filename, object &comp);

  /**
  * \param incarca textura in format TGA 4B necompresata de la 
  * adresa \param filename
  * in biectul \param comp
  */
  static void loadTexture(char* filename, object &comp);

  /**
  * \brief Deseneaza sky-box-ul care este o semisfera, nu este desenat
  * folosind metoda pentru obiecte normale deoarece locatia variaza neperiodic
  */
  static void drawSky();

  /**
  * \brief Deseneaza obiectul static \param comp
  */
  static void drawObject(object &comp);

  /**
  * \brief Creeaza vertex buffer object si indices buffer object
  * pe care ii incarca in memoria GPU, si seteaza id-urile din clasa object
  */
  static void setObjectId(object &comp);

  /**
  * \brief Deseneaza toate obiectele statice
  */
  void drawAllObjects();
};

/**
* \struct Move Reprezinta una din miscarile animatiei
* \param start reprezinta offset-ul in vectorul de
* frame-uri, iar length cate frame-uri are miscarea
*/
struct Move{
  int start, length;
};


/**
* \class MeshAnimation reprezinta un obiect animat
*/
class MeshAnimation{
public:

  /**
  * \param este un vector de mesh-uri care reprezinta frame-urile
  */
  std::vector<object> frames;

  /**
  * \param moves este vectorul cu miscarile disponibile
  */
  std::vector<Move> moves;

  /**
  * \param current_frame_no - frame-ul curent la care se afla animatia
  * in cadrul miscarii
  */
  int current_frame_no;

  /**
  * \param current_move - miscara curenta a animatiei
  */
  Move current_move;

  /**
  * \param filename - prefix-ul din nume comun tuturor frame-urilor
  * Frame-urile sunt citite din fisiere .obj
  * \param frame-no - numarul de frame-uri
  */
  MeshAnimation(const char* filename, int frame_no);
  ~MeshAnimation();
  /**
  * \brief Deseneaza frame-ul corect al animatiei
  */
  virtual void render(std::vector<object> &comp, float* buffer, float* restore_buffer);
};


/**
* \class MainCHaracter - clasa personajului urmarit de camera
*/
class MainCharacter : public MeshAnimation{
public:
  Camera *camera;
  /**
  * \param - position pozitia la care se afla personajul
  */
  Vector3 position;
  /**
  * \param orientation - unghiul in jurul axei y
  */
  float orientation;

  /**
  * \brief Constructor hardcodat - TODO generalizare
  */
  MainCharacter(const char* filename, Camera *camera);
  ~MainCharacter();

  /**
  * \brief deseneaza personajul animat
  */
  virtual void render(std::vector<object> &comp,  float* buffer, float* restore_buffer);

  /**
  * \brief reseteaza pozitia, rotatia personajului la cele initiale
  */
  void reset();

  /**
   * \brief Uncompleted - TODO
  */
  virtual void saveState();

  /**
  * \brief Salveaza starea curenta a jocului
  */
  virtual void loadState();

  /**
  * \brief Transforma bounding-box-ul intr-un vector de float-uri. A se vedea protocolul 
  * client- server 
  */
  virtual float* wrapBoundingBox(Vector3 &request_poz, int &inx, int id);

  /**
  * \brief ajusteaza pozitia personajului dupa rezultatele coliziunii - uncompleted TODO
  */
  void resolve_collision(float* wanted_pos, float* results);
  
  /**
  * \brief Trimite buffer-ul cu bounding-box-ul spre server
  */
  void sendBuffers(float *buffer_wanted, float* buffer_restore);
};


/**
* \class MovieIntro - clasa destinata reprezentarii unui filmulet
* pe o suprafata.
* Uncompleted TODO
* Problema principala: transformarea unui sir de imagini JPEG in imagini TGA
*/
class MovieIntro{
public:
  MovieIntro(char* filename, int frame_no);
  ~MovieIntro();
};
#endif