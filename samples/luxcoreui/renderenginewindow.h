/***************************************************************************
 * Copyright 1998-2018 by authors (see AUTHORS.txt)                        *
 *                                                                         *
 *   This file is part of LuxCoreRender.                                   *
 *                                                                         *
 * Licensed under the Apache License, Version 2.0 (the "License");         *
 * you may not use this file except in compliance with the License.        *
 * You may obtain a copy of the License at                                 *
 *                                                                         *
 *     http://www.apache.org/licenses/LICENSE-2.0                          *
 *                                                                         *
 * Unless required by applicable law or agreed to in writing, software     *
 * distributed under the License is distributed on an "AS IS" BASIS,       *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.*
 * See the License for the specific language governing permissions and     *
 * limitations under the License.                                          *
 ***************************************************************************/

#ifndef _LUXCOREAPP_RENDERENGINEWINDOW_H
#define	_LUXCOREAPP_RENDERENGINEWINDOW_H

#include "objecteditorwindow.h"
#include "typetable.h"
#include <memory>

class LuxCoreApp;

class RenderEngineWindow : public ObjectEditorWindow {
public:
	RenderEngineWindow(LuxCoreApp *a);
	virtual ~RenderEngineWindow() { }

	virtual void Open();

private:
	virtual void RefreshObjectProperties(const std::unique_ptr<luxrays::Properties> & props);
	virtual void ParseObjectProperties(const std::unique_ptr<luxrays::Properties> & props);
	virtual bool DrawObjectGUI(const std::unique_ptr<luxrays::Properties> & props, bool &modified);
	
	void DrawVarianceClampingSuggestedValue(const std::string &prefix,
			const std::unique_ptr<luxrays::Properties> & props, bool &modifiedProps);

        std::unique_ptr<luxrays::Properties> GetAllRenderEngineProperties(const std::unique_ptr<luxrays::Properties> & cfgProps) const;
	void PathGUI(const std::unique_ptr<luxrays::Properties> & props, bool &modifiedProps);
	void PathOCLGUI(const std::unique_ptr<luxrays::Properties> & props, bool &modifiedProps);
	void TilePathGUI(const std::unique_ptr<luxrays::Properties> & props, bool &modifiedProps);
	void TilePathOCLGUI(const std::unique_ptr<luxrays::Properties> & props, bool &modifiedProps);
	void BiDirGUI(const std::unique_ptr<luxrays::Properties> & props, bool &modifiedProps);
	void ThreadsGUI(const std::unique_ptr<luxrays::Properties> & props, bool &modifiedProps);

	TypeTable typeTable;
	float suggestedVerianceClampingValue;
};

#endif	/* _LUXCOREAPP_RENDERENGINEWINDOW_H */
