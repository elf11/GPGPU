#include "MenuGUI.h"

MenuGUI::MenuGUI()
{
  visible  = false;
  StaticObj::readObj("MenuBackground.bin", bkg_obj);
  StaticObj::readObj("menu_gui/logo.bin", logo);
  StaticObj::loadTexture("menu_gui/logo.tga", logo);

  object new_game_btn, new_game_dialog_box;
  StaticObj::readObj("menu_gui/standard_btn.bin", new_game_btn);
  StaticObj::loadTexture("menu_gui/resume_btn.tga", new_game_btn);
  StaticObj::readObj("menu_gui/dialog.bin", new_game_dialog_box);


  object load_game_btn = new_game_btn;
  object load_game_dialog = new_game_dialog_box;
  StaticObj::loadTexture("menu_gui/load_btn.tga", load_game_btn);

  object resume_game_btn = new_game_btn;
  object resume_game_dialog = new_game_dialog_box;
  StaticObj::loadTexture("menu_gui/newGame_btn.tga", resume_game_btn);

  object save_game_btn = new_game_btn;
  object salve_game_dialog = new_game_dialog_box;
  StaticObj::loadTexture("menu_gui/save_btn.tga", save_game_btn);

  object credits_game_btn = new_game_btn;
  object credits_game_dialog = new_game_dialog_box;
  StaticObj::loadTexture("menu_gui/credits_btn.tga", credits_game_btn);
  StaticObj::loadTexture("menu_gui/credits_dialog.tga", credits_game_dialog);

  object exit_game_btn = new_game_btn;
  object exit_game_dialog = new_game_dialog_box;
  StaticObj::loadTexture("menu_gui/exit_btn.tga", exit_game_btn);

  menu_list.push_back(new_game_btn);
  menu_list.push_back(load_game_btn);
  menu_list.push_back(resume_game_btn);
  menu_list.push_back(save_game_btn);
  menu_list.push_back(credits_game_btn);
  menu_list.push_back(exit_game_btn);

  dialog_list.push_back(new_game_dialog_box);
  dialog_list.push_back(load_game_dialog);
  dialog_list.push_back(resume_game_dialog);
  dialog_list.push_back(salve_game_dialog);
  dialog_list.push_back(credits_game_dialog);
  dialog_list.push_back(exit_game_dialog);

  screenWidth = 1280;
  screenHeight = 720;
  int selected = 0;
}

MenuGUI::~MenuGUI()
{
}

void MenuGUI::bindTexture()
{
  unsigned char *background;
  background = 
    (unsigned char*)malloc(sizeof(unsigned char)*4*screenWidth*screenHeight);
  glReadPixels(0, 0, screenWidth, screenHeight, GL_RGBA, GL_UNSIGNED_BYTE, background);
  selectiveBlur(background);
  glEnable(GL_TEXTURE_2D);
  glGenTextures( 1 , &bkg_obj.textureId );
	glBindTexture(GL_TEXTURE_2D , bkg_obj.textureId);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexImage2D(GL_TEXTURE_2D , 
			0, 
			4,
			screenWidth, 
			screenHeight, 
			0, 
			GL_RGBA, 
      GL_UNSIGNED_BYTE, 
			background);
  if(background != NULL){
    free(background);
  }
}

void MenuGUI::deleteTexture()
{
  glDeleteTextures(1, &bkg_obj.textureId);
}


void MenuGUI::display()
{
  glPushMatrix();
  glTranslatef(0, 0, -125);
  bkg_obj;
  StaticObj::drawObject(bkg_obj);

  
  glTranslatef(-20.0f, 0.0f, 0.0f);
  if(selected == 4){
      glTranslatef(40.0f, 0.0f, 0.0f);
      StaticObj::drawObject(dialog_list[4]);
      glTranslatef(-40.0f, 0.0f, 0.0f);
    }
  glTranslatef(0.0f, 5.0f, 0.0f);
  for(int i = 0; i < menu_list.size(); i++){
    if(i == selected){
      glTranslatef(10, 0, 0);
    }
    StaticObj::drawObject(menu_list[i]);
    if(i == selected){
      glTranslatef(-10, 0, 0);
    }

    glTranslatef(0.0f, -4, 0.0f);
  }
  glTranslatef(0.0f, 4.0f * menu_list.size(), 0.0f);
  glTranslatef(20.0f, 0.0f, 0.0f);

  glRotatef(30, 1.5f, 0, -1);
  glScalef(0.5f, 0.5f, 0.5f);
  glTranslatef(-45.0f, 25.0f, 60.0f);
  glRotatef(logo.rotation.y, 0, 1, 0);
  if(logo.rotation.y <= 360.0f)
    logo.rotation.y++;
  else logo.rotation.y -= 360.0f;
  
  StaticObj::drawObject(logo);
  glTranslatef(45.0f, -25.0f, -60.0f);
  glScalef(-0.5f, -0.5f, -0.5f);
  glTranslatef(0, 0, -125);
  glPopMatrix();
}

void MenuGUI::selectiveBlur(unsigned char *originalImg)
{
  for(int i = 0; i < screenHeight; i++){
    for(int j = 0; j < screenWidth ; j++){
      originalImg[i*screenWidth*4 + j*4+1] = originalImg[i*screenWidth*4 + j*4];
      originalImg[i*screenWidth*4 + j*4+2] = originalImg[i*screenWidth*4 + j*4];
    }
  }
}
