#include "keyStrokes.h"

extern void exitGame();

extern bool keys[256];
extern Camera *camera;
#define T_STEP 4.0f
#define R_STEP 2.0f

extern Vector3 translate_edit;
extern MenuGUI *menu;
extern MainCharacter * mainCharacter;

void KeyStrokes::keyPressed (unsigned char key, int x, int y)
{
  keys[key]  = true;
  switch(key){
    case 27: Escape_key();
      break;
    case 13: Enter_key();
      break;
    case 119: w_key();
      break;
    case 115: s_key();
      break;
    case 113 : mainCharacter->orientation = 270;
      break;
    case 101 : mainCharacter->orientation = 90;
      break;
  }
}

void KeyStrokes::w_key()
{
  if(menu->isVisible()){
    if(menu->selected == 0){
      menu->selected = menu->menu_list.size()-1;
    }
    else{
      menu->selected--;
    }
  }
  else{
    mainCharacter->orientation = 180;
  }
}

void KeyStrokes::s_key()
{
  if(menu->isVisible()){
    menu->selected = (menu->selected + 1)%menu->menu_list.size();
  }
  else{
    mainCharacter->orientation = 0;
  }
}

extern MainCharacter * mainCharacter;
void KeyStrokes::Enter_key()
{
  if(menu->isVisible()){
    switch(menu->selected){
      case 0: menu->hide();
        break;
      case 1: {mainCharacter->loadState();menu->hide();}
        break;
      case 2: {mainCharacter->reset();menu->hide();}
        break;
      case 3: mainCharacter->saveState();
        break;
      case 5: exitGame();
        break;
    }
  }
}

void KeyStrokes::Escape_key()
{
  if(menu->isVisible()){
    menu->hide();
  }
  else{
    menu->show();
  }
}

void KeyStrokes::keyUp (unsigned char key, int x, int y)
{
  keys[key]  = false;
  printf("%d\n", key);
}

void KeyStrokes::checkKey()
{
  if(keys[119] == true){
    camera->moveForward(T_STEP);
  }
  if(keys[115]){
    camera->moveBackward(T_STEP);
  }
  if(keys[113] == true){
    camera->moveLeft(T_STEP);
  }
  if(keys[101]){
    camera->moveRight(T_STEP);
  }

  if(keys[97] == true){
    camera->rotateLeft(R_STEP);
  }
  if(keys[100] == true){
    camera->rotateRight(R_STEP);
  }



  if(keys[105] == true){//i
    translate_edit.z++;
  }

  if(keys[106] == true){//j
    translate_edit.x++;
  }

  if(keys[107] == true){//k
    translate_edit.z--;
  }

  if(keys[108] == true){//l
    translate_edit.x--;
  }

  if(keys[109] == true){//m
    translate_edit.y--;
  }

  if(keys[110] == true){//n
    translate_edit.y++;
  }
    
}