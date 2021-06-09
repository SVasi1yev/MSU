#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <time.h>
#include <pwd.h>
#include <string.h>
#include <grp.h>

int printFileInfo(char* name0, char* realName, int argL)
{
	printf("--printFileInfo_start\n\n");
	
	if(argL)
	{
		struct stat st;
		if(lstat(name0, &st) == -1)
			return -1;

		char* access = malloc(12);
		for(int i = 0; i < 11; i++)
			access[i] = '0';
		access[3] = '-';
		access[7] = '-';
		access[11] = '\0';
		
		if(st.st_mode & S_IRUSR) access[0] = 'r';
		if(st.st_mode & S_IWUSR) access[1] = 'w';
		if(st.st_mode & S_IXUSR) access[2] = 'x';

		if (st.st_mode & S_IRGRP) access[4] = 'r';
		if (st.st_mode & S_IWGRP) access[5] = 'w';
		if (st.st_mode & S_IXGRP) access[6] = 'x';

		if (st.st_mode & S_IROTH) access[8] = 'r';
		if (st.st_mode & S_IWOTH) access[9] = 'w';
		if (st.st_mode & S_IXOTH) access[10] = 'x';
		
		char mode = 'n';
		
		if(S_ISDIR(st.st_mode))
			mode = 'd';
		else if(S_ISREG(st.st_mode))
			mode = '-';
		else if(S_ISCHR(st.st_mode))
			mode = 'c';
		else if(S_ISBLK(st.st_mode))
			mode = 'b';
		else if(S_ISFIFO(st.st_mode))
			mode = 'p';
		else if(S_ISLNK(st.st_mode))
			mode = 'l';
		
		struct passwd *userName;
		userName = getpwuid(st.st_uid);
		
		struct group *groupName;
		groupName = getgrgid(st.st_gid);
		
		struct tm *lastMod;
		lastMod = localtime(&(st.st_mtime));
					
		printf("name: %s\nmode: %c\naccess: %s\nlinks: %d\nsize: %d\nuser_name: %s\nuid: %d\ngroup_name: %s\ngid: %d\nlast_mod_data: %d-%d-%d %d.%d.%d\n\n", 
			   realName, mode, access, st.st_nlink, st.st_size, userName->pw_name, st.st_uid, groupName->gr_name, st.st_gid, 
			   lastMod->tm_hour, lastMod->tm_min, lastMod->tm_sec, lastMod->tm_mday, ((lastMod->tm_mon) + 1), ((lastMod->tm_year)%100));
		
		free(access);
	}
	else
	{
		printf("%s\n", realName);
	}
	
	printf("--printFileInfo_end\n\n");
	
	return 0;
}

int printDirInfo(char* name0, int argA, int argL, int argR)
{
	printf("--printDirInfo_start\n\n");
	
	DIR* dir = opendir(name0);
	if(dir == NULL)
		return -1;
	struct dirent *curF;
	struct stat st;
	char *name;
	char c = '\0';
	//char *realName;
	char **queue = malloc(sizeof(char*));
	int count = 0;
	
	printf("\n\n%s\n\n", name0);
	
	if((curF = readdir(dir)) == NULL)
	{
		free(queue);
		closedir(dir);
		return -1;
	}
	//realName = curF->d_name;
	while(curF != NULL)
	{
		if(((curF->d_name)[0] != '.') || argA)
		{
			name = (char*)malloc(strlen(name0) + strlen(curF->d_name) + 3);
			sprintf(name, "%s%s%c", name0, curF->d_name, c);
			//printf("%s\n", name);
			if(lstat(name, &st) == -1)
			{
				free(name);
				for (int i = 0; i < count; i++)
					free(queue[i]);
				free(queue);
				closedir(dir);
				return -1;
			}
			if(S_ISDIR(st.st_mode))
			{
				//printf("%i\n", strlen(name));
				name[strlen(name) + 1] = '\0';
				name[strlen(name)] = '/';
				if(printFileInfo(name, curF->d_name, argL) == -1)
				{
					free(name);
					for (int i = 0; i < count; i++)
						free(queue[i]);
					free(queue);
					closedir(dir);
					return -1;
				}
				//printf("%s\n", realName);
				if(argR && (strcmp(curF->d_name, ".") != 0) &&  (strcmp(curF->d_name, "..") != 0))
				{
					count++;
					queue = (char**)realloc(queue, count * sizeof(char*));
					queue[count - 1] = (char*)malloc(strlen(curF->d_name));
					strncpy(queue[count - 1], curF->d_name, strlen(curF->d_name) + 1);
				}
				free(name);
			}
			else
			{
				if(printFileInfo(name, curF->d_name, argL) == -1)
				{
					free(name);
					for (int i = 0; i < count; i++)
						free(queue[i]);
					free(queue);
					closedir(dir);
					return -1;
				}
				free(name);
			}
		}
		curF = readdir(dir);
		//realName = curF->d_name;
	}
	
	for(int i = 0; i < count; i++)
	{
		name = (char*)malloc(strlen(name0) + strlen(queue[i]) + 3);
		sprintf(name, "%s%s/%c", name0, queue[i], c);
		if(printDirInfo(name, argA, argL, argR) == -1)
		{
			for (int j = i; j < count; j++)
				free(queue[j]);
			free(queue);
			free(name);
			closedir(dir);
			return -1;
		}
		free(queue[i]);
		free(name);
	}
	
	free(queue);	
	closedir(dir);
	
	printf("--printDirInfo_end\n\n");
	
	return 0;	
}

int main(int argc, char* argv[])
{
	int argA = 0, argL = 0, argR = 0, fileNum = 0;
	struct stat st1;
	char c = '\0';
	
	printf("--Params_Check_Start\n");
	
	for(int i = 1; i < argc; i++)
	{
		if(*(argv[i]) == '-')
			switch(*(argv[i] + 1))
			{
				case 'a':
					argA = 1;
					break;
				case 'l':
					argL = 1;
					break;
				case 'R':
					argR = 1;
					break;
			}
		else 
		{
			fileNum = i;
			break;
		}
	}
	
	printf("argA = %i, argL = %i, argR = %i, fileNum = %i\n", argA, argL, argR, fileNum);
	
	printf("--Params_Check_End\n\n");
	
	if(fileNum == 0)
	{
		if(printDirInfo("./", argA, argL, argR) == -1)
		{
			perror("printDirInfo");
			return -1;
		}
	}
	else
	{
		lstat(argv[fileNum], &st1);
		
		if(S_ISDIR(st1.st_mode))
		{
			char *name;
			if((argv[fileNum])[strlen(argv[fileNum]) - 1] != '/')
			{
				name = malloc(strlen(argv[fileNum]) + 4);
				sprintf(name, "./%s/%c", argv[fileNum], c);
				if(printDirInfo(name, argA, argL, argR) == -1)
				{
					perror("printDirInfo");
					free(name);
					return -1;
				}
				free(name);
			}
			else
			{
				name = malloc(strlen(argv[fileNum]) + 3);
				sprintf(name, "./%s%c", argv[fileNum], c);
				if(printDirInfo(name, argA, argL, argR) == -1)
				{
					perror("printDirInfo");
					free(name);
					return -1;
				}
				free(name);
			}
		}
		else
		{
			char* realName = malloc(strlen(argv[fileNum]) + 1);
			int m = -1;
			for(int i = 0; argv[fileNum][i] != '\0'; i++)
				if(argv[fileNum][i] == '/')
					m = i;
			
			for(int i = m + 1; i <= strlen(argv[fileNum]); i++)
				realName[i - m - 1] = argv[fileNum][i];
			
			if(printFileInfo(argv[fileNum], realName, argL) == -1)
			{
				perror("printFileInfo");
				free(realName);
				return -1;
			}
			
			free(realName);
		}
	}
	
	return 0;
}
