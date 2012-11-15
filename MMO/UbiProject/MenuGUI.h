#ifndef _MENU_LIB_
#define _MENU_LIB_
#include <process.h>
#include "CompCore.h"
#include "staticObj.h"

/**
* \class MenuGUI - Clasa pentru meniul aplicatiei
* Contine butoane, cutii de dialog, background, logo
*/
class MenuGUI{
public:
  object bkg_obj;
  object logo;
  std::vector<object> menu_list;
  std::vector<object> dialog_list;
  int selected;
  volatile int screenWidth, screenHeight;
  bool visible;
  MenuGUI();
  ~MenuGUI();

  /**
  * \brief deseneaza meniul pe ecran
  */
  void display();

  /**
  * \brief Din cauza faptului ca se genereaza o textura noua
  * de fiecare data cand se intra in meniu, trebuie legata textura
  * noua de fiecare data
  */
  void bindTexture();

  /**
  * sterge textura dupa ce nu mai este folosita
  */
  void deleteTexture();

  /**
  * \brief Arata meni-ul, cam la fel ca si interfata din Java
  */
  inline void show()
  {
    visible = true;
    selected = 0;
    bindTexture();
  }
  /**
  * \brief Ascunde meniul sau mai bine zis impiedica desenarea
  * acestuia pe ecran
  */
  inline void hide() {visible = false;}
  
  /**
  * \return Returneaza daca sau nu meniul este vizibil
  */
  inline bool isVisible() const {return visible;}
  /**
  * \brief Calculeaza efectul imaginii de fundal
  */
  void selectiveBlur(unsigned char *originalImg);
};

#endif