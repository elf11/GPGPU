#include "staticObj.h"

//#pragma intrinsic(_strcmp)
//#pragma intrinsic(_abs)

extern int objNr_edit;
extern Vector3 translate_edit;
extern StaticObj *objects;
extern HighMap *logicMap;
StaticObj::StaticObj(const char* objFile)
{
  FILE *f = fopen(objFile, "r");
  if(f == NULL){
    std::cout<<"Error reading static object's main file";
    return;
  }
  char line[100];
  char filename[60];
  char textureFilename[60];
  
  while(fgets(line, 100, f)){
    if(line[0] != '#'){
      object comp;
      sscanf(line, "name=%s", comp.name);
      fgets(line, 100, f);
      sscanf(line, "filename=%s", filename);
      std::cout<<filename<<std::endl;
      fgets(line, 100, f);
      sscanf(line, "texture=%s", textureFilename);
      std::cout<<textureFilename<<std::endl;
      loadTexture(textureFilename, comp);
      readObj(filename, comp);
      fgets(line, 100, f);
      sscanf(line, "pozX=%f", &comp.location.x);
      fgets(line, 100, f);
      sscanf(line, "pozY=%f", &comp.location.y);
      fgets(line, 100, f);
      sscanf(line, "pozZ=%f", &comp.location.z);

      fgets(line, 100, f);
      sscanf(line, "angle=%f", &comp.angle);
      fgets(line, 100, f);
      sscanf(line, "rotX=%f", &comp.rotation.x);
      fgets(line, 100, f);
      sscanf(line, "rotY=%f", &comp.rotation.y);
      fgets(line, 100, f);
      sscanf(line, "rotZ=%f", &comp.rotation.z);

      fgets(line, 100, f);
      sscanf(line, "scaleX=%f", &comp.scaling.x);
      fgets(line, 100, f);
      sscanf(line, "scaleY=%f", &comp.scaling.y);
      fgets(line, 100, f);
      sscanf(line, "scaleZ=%f", &comp.scaling.z);
      fgets(line, 100, f);
      sscanf(line, "boundingbox=%s", filename);
      comp.setBoundingBox(filename);
      if(strcmp(comp.name, "sentinel") == 0){
        Vector3 aux_loc;
        aux_loc.x = 303.0f;
        aux_loc.y = -6.0f;
        aux_loc.z = -388.0f;
        comp.duplicates.push_back(aux_loc);
      }
      if(strcmp(comp.name, "grass") == 0){
        Vector3 aux_loc;

        for(int i = 0; i < 100; i++){
          aux_loc.x = (rand() % 1000) - 500;
          aux_loc.z = (rand() % 800) - 400;
          comp.duplicates.push_back(aux_loc);
        }
        
      }
      components.push_back(comp);
    }
    if(line[0] == '\n')
      break;
  }
  fclose(f);
}

StaticObj::~StaticObj()
{
}

void StaticObj::readObj(char* filename, object &comp)
{
  FILE *f = fopen(filename, "rb");
  if(f == NULL){
    std::cout<<"Error opening an object file";
    return;
  }
  int length;
  fread(&length, sizeof(int), 1, f);
  std::vector<Vector3> vertices(length);
  fread(&vertices[0], sizeof(Vector3)*length, 1, f);

  fread(&length, sizeof(int), 1, f);
  std::vector<Vector3> normals(length);
  fread(&normals[0], sizeof(Vector3)*length, 1, f);

  fread(&length, sizeof(int), 1, f);
  if(length >0){
    std::vector<Point2D> texMap(length);
    fread(&texMap[0], sizeof(Point2D)*length, 1, f);
  
    fread(&length, sizeof(int), 1, f);
    std::vector<Face> face(length);
    fread(&face[0], sizeof(Face)*length, 1, f);
    comp.texMap = texMap;
    comp.face = face;
  }

  comp.vertices = vertices;
  comp.normals = normals;
  fclose(f);
  setObjectId(comp);
}

void StaticObj::setObjectId(object &comp)
{
  GLfloat *v = new GLfloat[comp.face.size() * sizeof(float) * 8 * 3];

  for(int i = 0; i < comp.face.size(); i++){
    v[i*24] = comp.vertices[comp.face[i].v.x].x;
    v[i*24+1] = comp.vertices[comp.face[i].v.x].y;
    v[i*24+2] = comp.vertices[comp.face[i].v.x].z;
    v[i*24+3] = comp.normals[comp.face[i].n.x].x;
    v[i*24+4] = comp.normals[comp.face[i].n.x].y;
    v[i*24+5] = comp.normals[comp.face[i].n.x].z;
    v[i*24+6] = comp.texMap[comp.face[i].uv.x].x;
    v[i*24+7] = comp.texMap[comp.face[i].uv.x].y;

    v[i*24+8] = comp.vertices[comp.face[i].v.y].x;
    v[i*24+9] = comp.vertices[comp.face[i].v.y].y;
    v[i*24+10] = comp.vertices[comp.face[i].v.y].z;
    v[i*24+11] = comp.normals[comp.face[i].n.y].x;
    v[i*24+12] = comp.normals[comp.face[i].n.y].y;
    v[i*24+13] = comp.normals[comp.face[i].n.y].z;
    v[i*24+14] = comp.texMap[comp.face[i].uv.y].x;
    v[i*24+15] = comp.texMap[comp.face[i].uv.y].y;

    v[i*24+16] = comp.vertices[comp.face[i].v.z].x;
    v[i*24+17] = comp.vertices[comp.face[i].v.z].y;
    v[i*24+18] = comp.vertices[comp.face[i].v.z].z;
    v[i*24+19] = comp.normals[comp.face[i].n.z].x;
    v[i*24+20] = comp.normals[comp.face[i].n.z].y;
    v[i*24+21] = comp.normals[comp.face[i].n.z].z;
    v[i*24+22] = comp.texMap[comp.face[i].uv.z].x;
    v[i*24+23] = comp.texMap[comp.face[i].uv.z].y;
  }
  glGenBuffers(1, &comp.VBOID);
  glBindBuffer(GL_ARRAY_BUFFER, comp.VBOID);
  glBufferData(GL_ARRAY_BUFFER, comp.face.size() * sizeof(float) * 8 * 3, v, GL_STATIC_DRAW);
  delete[] v;

  GLuint *indicies = new GLuint[comp.face.size()*sizeof(GLuint)*3*3];
  for(int i = 0; i < comp.face.size()*3; i++){
    indicies[i] = i;
  }
  glGenBuffers(1, &comp.IBOID);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, comp.IBOID);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, comp.face.size()*3*sizeof(Vector3Int), indicies, GL_STATIC_DRAW);
  delete[] indicies;
}

void StaticObj::loadTexture(char* filename, object &comp)
{
  TGAImage * pImagineTextura = new TGAImage(filename);
	GLuint text=0;
	if(pImagineTextura != NULL){
    glEnable(GL_TEXTURE_2D);
		glGenTextures( 1 , &text );
		glBindTexture(GL_TEXTURE_2D , text);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D , 
			0, 
			4,
			pImagineTextura->width, 
			pImagineTextura->height, 
			0, 
			GL_RGBA, 
      GL_UNSIGNED_BYTE, 
			pImagineTextura->data);
	}
	if( pImagineTextura ){
		if(pImagineTextura->data){
      free (pImagineTextura->data);
    }
		free(pImagineTextura);
	}
  comp.textureId = text;
}

void StaticObj::drawSky()
{
    glPushMatrix();
      Vector3::translate(objects->components[0].location);
      Vector3::rotate(objects->components[0].rotation, objects->components[0].angle);
      StaticObj::drawObject(objects->components[0]);
      Vector3::rotateBack(objects->components[0].rotation, objects->components[0].angle);
      Vector3::translateBack(objects->components[0].location);
    glPopMatrix();
}

void StaticObj::drawAllObjects()
{
  Vector3 aux;

  glPushMatrix();
  for(int i = 1; i < components.size(); i++){
    if(i == objNr_edit){
      aux = translate_edit;
      std::cout<<translate_edit.x<<" "<<translate_edit.y<<" "<<translate_edit.z<<std::endl;
    }
    if(strcmp(components[i].name, "Propeller") == 0){
      if(components[i].angle > 360)
        components[i].angle -= 360.0f;
      components[i].angle++;
      glPushMatrix();
    }
    if(strcmp(components[i].name , "helicopter_body") == 0){
      Vector3 translation;
      translation.y = 13.073f;
      Vector3::translate(translation);
      glPushMatrix();
    }
    if(strcmp(components[i].name, "helicopter_main_propeller") == 0){

      if(components[i].angle > 360)
        components[i].angle -= 360.0f;
      components[i].angle += 20;
      glPushMatrix();
    }
    if(strcmp(components[i].name, "helicopter_second_propeller") == 0){
      if(components[i].angle > 360)
        components[i].angle -= 360.0f;
      components[i].angle += 20;
      Vector3 translation;
      translation.z = 65.229f;
      translation.y = 13.073f;
      Vector3::translate(translation);
      glPushMatrix();
    }
    glEnable(GL_ALPHA_TEST);
    glAlphaFunc (GL_GREATER, 0.5f);
    
    for(int j = 0; j < components[i].duplicates.size(); j++){
      Vector3::translate(components[i].duplicates[j]);
      drawObject(components[i]);
      Vector3::translateBack(components[i].duplicates[j]);
    }

    Vector3::translate(components[i].location + aux);
    Vector3::rotate(components[i].rotation, components[i].angle);
    drawObject(components[i]);
    Vector3::rotateBack(components[i].rotation, components[i].angle);
    Vector3::translateBack(components[i].location + aux);
    

    glDisable(GL_ALPHA_TEST);

    if(strcmp(components[i].name, "helicopter_main_propeller") == 0){
      glPopMatrix();
    }
    if(strcmp(components[i].name, "helicopter_second_propeller") == 0){
      glPopMatrix();
      Vector3 translation;
      translation.z = 65.229f;
      translation.y = 13.073f;
      Vector3::translateBack(translation);
      glPopMatrix();
      translation.y = 13.073f;
      Vector3::translateBack(translation);
    }
    if(strcmp(components[i].name, "Propeller") == 0){
      glPopMatrix();
    }
  }
  glPopMatrix();
}

void StaticObj::drawObject(object &comp)
{
  glBindTexture(GL_TEXTURE_2D, comp.textureId);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, comp.textureId);
  glEnable(GL_TEXTURE_2D);
  glBindBuffer(GL_ARRAY_BUFFER, comp.VBOID);
  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(3, GL_FLOAT, 8*sizeof(GLfloat), (void*)0);
  glEnableClientState(GL_NORMAL_ARRAY);
  glNormalPointer(GL_FLOAT, 8*sizeof(GLfloat), (void*)12);
  glClientActiveTexture(GL_TEXTURE0);
  glEnableClientState(GL_TEXTURE_COORD_ARRAY);
  glTexCoordPointer(2, GL_FLOAT, 8*sizeof(GLfloat), (void*)24);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, comp.IBOID);
  glDrawElements(GL_TRIANGLES, comp.face.size()*3*3, GL_UNSIGNED_INT, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glDisable(GL_TEXTURE_2D);
}


MeshAnimation::MeshAnimation(const char *filename, int frame_no)
{
  object aux_obj;
  char complete_file_name[60];
  strcpy(complete_file_name, filename);
  strcat(complete_file_name, "1");
  strcat(complete_file_name, ".bin");

  StaticObj::readObj(complete_file_name, aux_obj);
  StaticObj::loadTexture("static_objects/freya_texture.tga", aux_obj);
  aux_obj.setBoundingBox("static_objects/freya.bb");
  aux_obj.bounding_box[0]->type = BoundingBox::PARALLELEPIPED;
  frames.push_back(aux_obj);
  
  for(int i = 1; i < frame_no; i++){
    strcpy(complete_file_name, filename);
    char nr[10];
    itoa(i+1, nr, 10);
    strcat(complete_file_name, nr);
    strcat(complete_file_name, ".bin");
    object aux_obj2;
    aux_obj2.textureId = aux_obj.textureId;
    StaticObj::readObj(complete_file_name, aux_obj2);
    aux_obj2.setBoundingBox("static_objects/freya.bb");
    frames.push_back(aux_obj2);
    frames[i].vertices = aux_obj2.vertices;
    frames[i].normals = aux_obj2.normals;
  }
}

MeshAnimation::~MeshAnimation()
{
}

void MeshAnimation::render(std::vector<object> &comp,  float* buffer, float* restore_buffer)
{
  int inx = 
    (current_frame_no + current_move.start)%current_move.length;
  StaticObj::drawObject(comp[inx]);
  current_frame_no++;
}

MainCharacter::MainCharacter(const char* filename, Camera *camera): MeshAnimation(filename, 28)
{
  this->camera = camera;
  Move move;
  move.start = 0;
  move.length = 28;
  moves.push_back(move);
  current_frame_no = 0;
  current_move = moves[0];
  position.y = 0;
  position.z = 0;
  position.x = 0;
  orientation = 180;
}

MainCharacter::~MainCharacter()
{
}

void MainCharacter::render(std::vector<object> &comp, float* buffer, float* restore_buff)
{ 
  for(int i = 0; i < buffer[0]; i++){
  }
  
  //resolve_collision(buffer, restore_buff);
  if(restore_buff[2] == buffer[2] && restore_buff[3] == buffer[3] && restore_buff[4] == buffer[4]){
    glTranslatef(camera->undo_translation.x - camera->translation.x, camera->undo_translation.y - camera->translation.y, camera->undo_translation.z - camera->translation.z);
    (camera->undo_translation - camera->translation).print();
    position = camera->undo_translation;
    camera->translation = camera->undo_translation;
  }
  
  position.x = -position.x;
  position.z = -position.z-100;
  glTranslatef(position.x, position.y, position.z);
  glRotatef(orientation-camera->rotation.y, 0, 1, 0);
  int inx = (current_frame_no + current_move.start)%current_move.length;
  StaticObj::drawObject(comp[inx]);

  glRotatef(-orientation+camera->rotation.y, 0, 1, 0);
  glTranslatef(-position.x, -position.y, -position.z);
  current_frame_no++;
}

extern StaticObj *objects;

void MainCharacter::resolve_collision(float* wanted_pos, float* restore)
{
  for(int i = 0; i < wanted_pos[0]; i++){
    if(wanted_pos[i*7+1] == 0){
      AAParallelepiped *p
        = new AAParallelepiped(wanted_pos[i*7+2],
                              wanted_pos[i*7+3],
                              wanted_pos[i*7+4],
                              wanted_pos[i*7+5],
                              wanted_pos[i*7+6],
                              wanted_pos[i*7+7]);
      for(int j = 0; j < objects->components.size(); j++){
        for(int k = 0; k < objects->components[j].bounding_box.size(); k++){
          if(objects->components[j].bounding_box[k]->getType() == BoundingBox::PARALLELEPIPED){
            AAParallelepiped *pp = reinterpret_cast<AAParallelepiped*>(objects->components[j].bounding_box[k]);
            AAParallelepiped *p2  = new AAParallelepiped(pp->minPoint + objects->components[j].location, 
                                                         pp->maxPoint + objects->components[j].location);
            objects->components[j].location.print();
            p2->type = BoundingBox::PARALLELEPIPED;
            if(p->colide(p2)){
              printf("Colide\n");
              wanted_pos[i*7+2] = restore[i*7+2];
              wanted_pos[i*7+3] = restore[i*7+3];
              wanted_pos[i*7+4] = restore[i*7+4];
              wanted_pos[i*7+5] = restore[i*7+5];
              wanted_pos[i*7+6] = restore[i*7+6];
              wanted_pos[i*7+7] = restore[i*7+7];
              break;
            }
            delete p2;
          }
        }
      }
    }
  }
}



float* MainCharacter::wrapBoundingBox(Vector3 &request_poz, int &inx, int id)
{
  float *buffer = new float[BUFF_SIZE];
  //buffer[0] = this->frames[0].bounding_box.size();
  inx = 0;
  buffer[inx++] = id;
  buffer[inx++] = -1;
  for(int i = 0; i < this->frames[0].bounding_box.size(); i++){
    if(this->frames[0].bounding_box[i]->getType() == BoundingBox::PARALLELEPIPED){
      AAParallelepiped *p = 
        reinterpret_cast<AAParallelepiped*>(this->frames[0].bounding_box[i]);
      //buffer[inx++] = 0;
      buffer[inx++] = p->minPoint.x + request_poz.x;
      buffer[inx++] = p->minPoint.y + request_poz.y;
      buffer[inx++] = p->minPoint.z + request_poz.z;
      buffer[inx++] = p->maxPoint.x + request_poz.x;
      buffer[inx++] = p->maxPoint.y + request_poz.y;
      buffer[inx++] = p->maxPoint.z + request_poz.z;
    }
    
    if(this->frames[0].bounding_box[i]->getType() == BoundingBox::SPHERE){
      //buffer[inx++] = 1;
      Sphere *s = reinterpret_cast<Sphere*>(this->frames[0].bounding_box[i]);
      buffer[inx++] = this->frames[0].location.x + request_poz.x;
      buffer[inx++] = this->frames[0].location.y + request_poz.y;
      buffer[inx++] = this->frames[0].location.z + request_poz.z;
      buffer[inx++] = s->radius;
    }
    
  }
  
  return buffer;
}

void MainCharacter::reset()
{
  position.y = 0;
  position.z = 0;
  position.x = 0;
  camera->reset();
}

void MainCharacter::saveState()
{
  /*
  printf("State saved\n");
  FILE *f = fopen("save.sav", "w");
  fprintf(f, "%f %f %f\n%f %f %f\n%f %f %f", 
    position.x, 
    position.y, 
    position.z,
    camera->rotation.x,
    camera->rotation.y,
    camera->rotation.z,
    camera->translation.x, 
    camera->translation.y, 
    camera->translation.z);
  fclose(f);
  */
}

void MainCharacter::loadState()
{
  /*
  printf("State loaded\n");
  FILE *f = fopen("save.sav", "r");
  fscanf(f, "%f %f %f\n%f %f %f", 
    &position.x,
    &position.y,
    &position.z,
    &camera->rotation.x,
    &camera->rotation.y,
    &camera->rotation.z,
    &camera->translation.x, 
    &camera->translation.y, 
    &camera->translation.z);
  fclose(f);
  */
}